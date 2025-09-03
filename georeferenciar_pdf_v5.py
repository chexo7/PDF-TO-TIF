# -*- coding: utf-8 -*-
"""
GeoRef PDF → GeoTIFF (PyQt6, 3 GCPs Affine)
• Robust rasterization (DPI + Max MP + markups)
• Wheel zoom, pan, crosshair cursor
• Visible, draggable markers (max 3)
• Left-click = add point, right-click = delete nearest
• GCP input in 2 modes:
    - Lat/Lon (EPSG:4326)
    - X/Y (State Plane NAD83) with EPSG dropdown and automatic conversion to Lon/Lat
• Exact affine (triangulation, includes rotation/scale) → GeoTIFF (EPSG:4326)

NEW:
    - No lossy JPG intermediate: rasterizes PDF → NumPy array (lossless from PyMuPDF)
    - Output options:
        * JPEG (100%, YCbCr)
        * DEFLATE (8-bit Grayscale)
        * DEFLATE (Color)
        * JPEG2000
"""

import os
import sys
import math
import tempfile
import logging
import traceback
from datetime import datetime

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt

import fitz  # PyMuPDF
import rasterio
from rasterio.transform import Affine

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QFormLayout, QStatusBar, QComboBox,
    QGraphicsEllipseItem, QGraphicsSimpleTextItem
)

# --- pyproj for CRS and transforms ---
from pyproj import CRS, Transformer
try:
    from pyproj.database import query_crs_info
except Exception:
    query_crs_info = None

__author__ = "Sergio Acevedo - GHD"

# --------------------------- LOGGING ---------------------------------
LOG = logging.getLogger("georef")
LOG.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
LOG.addHandler(_console_handler)

_LOG_DIR = tempfile.gettempdir()
_LOG_FILE = os.path.join(_LOG_DIR, f"georef_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
LOG.addHandler(_file_handler)

# Excepthook: capture unhandled exceptions WITHOUT closing the app
_def_excepthook = sys.excepthook
def _log_excepthook(exc_type, exc_value, exc_tb):
    try:
        tb_txt = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        LOG.critical("Unhandled exception:\n%s", tb_txt)
        try:
            app = QApplication.instance()
            if app is not None:
                msg = "An unexpected error occurred.\n\n" + tb_txt[:1500]
                QMessageBox.critical(None, "Unhandled error", msg)
        except Exception:
            pass
    except Exception:
        pass
    # Do not call default excepthook (keeps process alive).
sys.excepthook = _log_excepthook

# --------------------------- Rasterization ----------------------------
class RasterizeError(Exception):
    pass

def _compute_scales_for_limits(rect_width_pt, rect_height_pt, dpi, max_megapixels=120.0):
    """
    Compute effective scale to satisfy the Max MP constraint.
    Returns (eff_scale, out_w_px, out_h_px, eff_dpi)
    """
    base_scale = max(dpi, 72) / 72.0
    w_px = rect_width_pt * base_scale
    h_px = rect_height_pt * base_scale

    # cap by megapixels
    mp = (w_px * h_px) / 1e6
    eff_scale = base_scale
    if mp > max_megapixels:
        factor_mp = math.sqrt(max_megapixels / mp)
        eff_scale *= factor_mp
        LOG.warning("Auto downscale by MP: factor=%.3f (target ≤ %.1f MP)", factor_mp, max_megapixels)

    # final dims & dpi
    w_px = rect_width_pt * eff_scale
    h_px = rect_height_pt * eff_scale
    eff_dpi = 72.0 * eff_scale
    return eff_scale, int(round(w_px)), int(round(h_px)), eff_dpi

def pdf_page_to_array_safe(pdf_path, page_index, dpi, max_megapixels=120.0,
                           include_annots=True):
    """
    Rasterize a PDF page to a NumPy array (H, W, 3) in RGB, losslessly from PyMuPDF.
    Applies guard for Max MP.
    Returns (rgb_array_uint8, eff_dpi)
    """
    LOG.info("Rasterizing PDF: %s | page: %s | dpi: %s | maxMP: %.1f | annots: %s",
             pdf_path, page_index + 1, dpi, max_megapixels, include_annots)

    if not os.path.isfile(pdf_path):
        raise RasterizeError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RasterizeError(f"Could not open PDF: {e}")

    try:
        if not (0 <= page_index < len(doc)):
            raise RasterizeError(f"Invalid page index. PDF has {len(doc)} pages.")
        page = doc.load_page(page_index)
        rect = page.rect  # in points (1/72 in)
        LOG.debug("PDF size (pt): width=%.2f, height=%.2f", rect.width, rect.height)

        eff_scale, out_w, out_h, eff_dpi = _compute_scales_for_limits(
            rect.width, rect.height, dpi, max_megapixels
        )
        LOG.debug("Final raster size: %d x %d px (eff_dpi=%.1f)", out_w, out_h, eff_dpi)

        mat = fitz.Matrix(eff_scale, eff_scale)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False, annots=include_annots)
        except TypeError:
            pix = page.get_pixmap(matrix=mat, alpha=False)
        w, h, n = pix.width, pix.height, pix.n
        if n not in (3, 4, 1, 2):
            LOG.warning("Unexpected number of channels from PyMuPDF: %d; forcing RGB", n)

        # Build RGB array
        samples = pix.samples  # bytes
        arr = np.frombuffer(samples, dtype=np.uint8).reshape(h, w, n)
        if n == 4:
            # ignore alpha, convert to RGB
            arr = arr[:, :, :3]
        elif n == 1 or n == 2:
            # grayscale → RGB
            gray = arr[:, :, 0]
            arr = np.stack([gray, gray, gray], axis=-1)
        # ensure C-order contiguous
        arr = np.ascontiguousarray(arr)
        return arr, eff_dpi
    except Exception as e:
        tb = traceback.format_exc()
        raise RasterizeError(f"Rasterization failed: {e}\nTraceback:\n{tb}")
    finally:
        doc.close()

# --------------------------- Geo-transform ---------------------------
def compute_affine_transform(colrow_pts, lons, lats):
    """
    Exact affine transform with 3 GCPs: (col,row) → (lon,lat).
      lon = a*col + b*row + c
      lat = d*col + e*row + f
    Returns: (Affine, rms_lon, rms_lat, angle_deg)
    """
    cols = np.array([p[0] for p in colrow_pts], dtype=float)
    rows = np.array([p[1] for p in colrow_pts], dtype=float)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    if len(cols) != 3:
        raise ValueError("Exactly 3 points are required for triangulation.")

    M = np.column_stack([cols, rows, np.ones_like(cols)])

    params_lon = np.linalg.solve(M, lons)
    params_lat = np.linalg.solve(M, lats)
    a, b, c = params_lon
    d, e, f = params_lat

    lons_pred = M @ params_lon
    lats_pred = M @ params_lat
    rms_lon = float(np.sqrt(np.mean((lons_pred - lons) ** 2)))
    rms_lat = float(np.sqrt(np.mean((lats_pred - lats) ** 2)))

    ang_rad = math.atan2(d, a)  # direction of image x-axis in (lon,lat)
    ang_deg = math.degrees(ang_rad)

    transform = Affine(a, b, c, d, e, f)
    return transform, rms_lon, rms_lat, ang_deg

def write_geotiff_from_array(rgb_array, transform, out_tif_path, crs_epsg=4326,
                             codec='jpeg', jpeg_quality=100, grayscale=False,
                             tile_size=512):
    """
    Write a GeoTIFF using the given transform and CRS.
    codec: 'jpeg', 'deflate' or 'jpeg2000'
    grayscale: if True and codec == 'deflate', writes 1-band 8-bit (MINISBLACK)
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb_array must be HxWx3 uint8.")
    if rgb_array.dtype != np.uint8:
        raise ValueError("rgb_array must be uint8.")

    H, W, _ = rgb_array.shape

    # Convert to grayscale if requested
    if grayscale and codec == 'deflate':
        # ITU-R BT.601 luma
        gray = (0.299 * rgb_array[:, :, 0] +
                0.587 * rgb_array[:, :, 1] +
                0.114 * rgb_array[:, :, 2]).astype(np.uint8)
        count = 1
        data = gray[np.newaxis, :, :]  # (1, H, W)
        photometric = 'MINISBLACK'
        dtype = gray.dtype
    else:
        # 3-band RGB
        data = np.transpose(rgb_array, (2, 0, 1))  # (3, H, W)
        count = 3
        dtype = rgb_array.dtype
        # photometric depends on codec
        photometric = 'YCbCr' if codec == 'jpeg' else 'RGB'

    profile = {
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': count,
        'dtype': dtype,
        'transform': transform,
        'crs': f"EPSG:{crs_epsg}",
        'tiled': True,
        'blockxsize': tile_size,
        'blockysize': tile_size,
        'interleave': 'pixel',
        'photometric': photometric
    }

    if codec == 'jpeg':
        profile.update({
            'compress': 'jpeg',
            'jpeg_quality': int(jpeg_quality),
        })
    elif codec == 'deflate':
        profile.update({
            'compress': 'deflate',
            'predictor': 2,  # horizontal differencing
            'zlevel': 7
        })
    elif codec == 'jpeg2000':
        profile.update({
            'compress': 'jpeg2000'
        })
    else:
        raise ValueError("codec must be 'jpeg', 'deflate' or 'jpeg2000'.")

    with rasterio.open(out_tif_path, 'w', **profile) as dst:
        dst.write(data)

    LOG.info("GeoTIFF exported → %s (EPSG:%s, codec=%s, grayscale=%s)",
             out_tif_path, crs_epsg, codec, grayscale)
    return out_tif_path

# --------------------------- Marker items ----------------------------
class MarkerItem(QGraphicsEllipseItem):
    """Movable circular marker, constant on-screen size, with label."""
    def __init__(self, index, x, y, img_w, img_h, move_callback, radius=6.0, parent=None):
        super().__init__(-radius, -radius, radius*2, radius*2, parent)
        self.setBrush(Qt.GlobalColor.red)
        self.setPen(Qt.GlobalColor.white)
        self.setZValue(10_000)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self.index = index
        self.img_w = img_w
        self.img_h = img_h
        self.move_callback = move_callback
        self.setPos(QPointF(x, y))

        self.label = QGraphicsSimpleTextItem(str(index + 1), self)
        self.label.setBrush(Qt.GlobalColor.yellow)
        self.label.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.label.setPos(radius + 2, -radius - 2)
        self.setToolTip(f"Point #{index+1} (drag me)")

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange:
            new_pos = value
            x = max(0.0, min(self.img_w - 1.0, new_pos.x()))
            y = max(0.0, min(self.img_h - 1.0, new_pos.y()))
            return QPointF(x, y)
        elif change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            if callable(self.move_callback):
                pos = self.pos()
                self.move_callback(self.index, float(pos.x()), float(pos.y()))
        return super().itemChange(change, value)

    def update_index(self, new_index):
        self.index = new_index
        self.label.setText(str(new_index + 1))
        self.setToolTip(f"Point #{new_index+1} (drag me)")

# --------------------------- Image view ------------------------------
class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.image_size = None

        self.left_click_callback = None     # add point
        self.right_click_callback = None    # delete point
        self.move_callback = None           # show coords

        self._zoom = 1.0
        self.setCursor(Qt.CursorShape.CrossCursor)

    def reset_view(self):
        self.resetTransform()
        self._zoom = 1.0
        if self.scene is not None:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_image(self, qpix: QPixmap):
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(qpix)
        self.image_size = (qpix.width(), qpix.height())
        self.setSceneRect(0, 0, qpix.width(), qpix.height())
        self.reset_view()

    def wheelEvent(self, event):
        if self.pixmap_item is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        new_zoom = self._zoom * factor
        if 0.05 <= new_zoom <= 50.0:
            self._zoom = new_zoom
            self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        if self.pixmap_item is not None and self.move_callback:
            pt_scene = self.mapToScene(event.position().toPoint())
            x, y = pt_scene.x(), pt_scene.y()
            if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                self.move_callback(float(x), float(y))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.pixmap_item is None:
            return super().mousePressEvent(event)
        pt_scene = self.mapToScene(event.position().toPoint())
        x, y = pt_scene.x(), pt_scene.y()
        if not (0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]):
            return super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self.left_click_callback:
            self.left_click_callback(float(x), float(y))
        elif event.button() == Qt.MouseButton.RightButton and self.right_click_callback:
            self.right_click_callback(float(x), float(y))
        super().mousePressEvent(event)

# --------------------------- Main window -----------------------------

class GeoRefApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoRef PDF → GeoTIFF (3 GCPs, Rot/Scale) + Zoom/Drag + SPCS NAD83")
        self.resize(1360, 930)

        # State
        self.pdf_path = None
        self.page_count = 0
        self.transform = None

        # Image cache (to avoid GC)
        self._rgb_array = None     # NumPy array HxWx3
        self._pil_img = None
        self._qimage = None
        self._pixmap = None
        self.img_w = 0
        self.img_h = 0

        # Markers
        self.markers = []  # list[MarkerItem]

        # CRS / mode
        self.input_mode = "latlon"  # "latlon" | "xy"
        self.selected_epsg = None   # NAD83 State Plane EPSG when input_mode == "xy"

        # For preserving points on re-rasterize (same PDF/page)
        self._last_pdf_for_raster = None
        self._last_page_idx_for_raster = None

        # UI
        self._build_ui()
        self._build_menu_toolbar()
        self.setStatusBar(QStatusBar())

        # View callbacks
        self.view.left_click_callback = self._handle_left_click
        self.view.right_click_callback = self._handle_right_click
        self.view.move_callback = self._handle_mouse_move

        # Populate SPCS NAD83
        self._populate_stateplane_combo()

        LOG.info("==== GeoRef GUI started ====")

    # ------------------- UI -------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Left panel: image
        self.view = ImageView()
        main.addWidget(self.view, stretch=3)

        # Right panel: controls
        right = QVBoxLayout()
        main.addLayout(right, stretch=2)

        # Group: PDF → Image
        grp_pdf = QGroupBox("PDF → Image")
        f1 = QFormLayout(grp_pdf)

        self.lbl_pdf = QLabel("PDF: (not loaded)")
        self.btn_pdf = QPushButton("Select PDF…")
        self.btn_pdf.clicked.connect(self._choose_pdf)

        # Raster params
        self.spn_page = QSpinBox(); self.spn_page.setMinimum(1); self.spn_page.setMaximum(1); self.spn_page.setValue(1)
        self.spn_dpi = QSpinBox(); self.spn_dpi.setRange(72, 1200); self.spn_dpi.setValue(300)
        self.spn_maxmp = QSpinBox(); self.spn_maxmp.setRange(10, 600); self.spn_maxmp.setValue(120)  # MP
        self.chk_ann = QCheckBox("Include annotations/markups"); self.chk_ann.setChecked(True)

        self.btn_raster = QPushButton("Rasterize page → Image")
        self.btn_raster.clicked.connect(self._rasterize)

        f1.addRow(self.lbl_pdf, self.btn_pdf)
        f1.addRow(QLabel("Page"), self.spn_page)
        f1.addRow(QLabel("DPI"), self.spn_dpi)
        f1.addRow(QLabel("Max Megapixels"), self.spn_maxmp)
        f1.addRow(self.chk_ann)
        f1.addRow(self.btn_raster)
        right.addWidget(grp_pdf)

        # Group: GCP input mode & Projection
        grp_mode = QGroupBox("GCP coordinates (input)")
        fmode = QFormLayout(grp_mode)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Lat/Lon (EPSG:4326)", "X/Y (State Plane NAD83)"])
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.cmb_spcs = QComboBox()
        self.cmb_spcs.setEditable(True)
        self.cmb_spcs.setEnabled(False)
        self.lbl_epsg = QLabel("EPSG: —")

        fmode.addRow(QLabel("Mode"), self.cmb_mode)
        fmode.addRow(QLabel("SPCS NAD83"), self.cmb_spcs)
        fmode.addRow(QLabel("Selected EPSG"), self.lbl_epsg)
        right.addWidget(grp_mode)

        # Group: GCPs (3 points)
        grp_gcps = QGroupBox("GCPs: 3 points (Triangulation)")
        v_gcps = QVBoxLayout(grp_gcps)

        self.tbl = QTableWidget(0, 4)
        self._set_table_headers_for_latlon()
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(self.tbl.EditTrigger.AllEditTriggers)

        hint = QLabel("Left click: add point (max 3). Right click: delete nearest. "
                      "Drag markers to refine.")
        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.clicked.connect(self._clear_points)

        v_gcps.addWidget(hint)
        v_gcps.addWidget(self.tbl)
        v_gcps.addWidget(self.btn_clear)
        right.addWidget(grp_gcps)

        # Group: Transform & Output
        grp_out = QGroupBox("Transform & Output")
        v_out = QVBoxLayout(grp_out)

        self.btn_compute = QPushButton("Compute transform (includes rotation & scale)")
        self.btn_compute.clicked.connect(self._compute_transform)

        # Output options
        opt_layout = QFormLayout()
        self.cmb_codec = QComboBox()
        self.cmb_codec.addItems([
            "JPEG (100%, YCbCr)",
            "DEFLATE (8-bit Grayscale)",
            "DEFLATE (Color)",
            "JPEG2000"
        ])
        self.cmb_codec.currentIndexChanged.connect(self._update_export_info)

        opt_layout.addRow(QLabel("Compression"), self.cmb_codec)

        self.lbl_result = QLabel("Result: —")
        self.lbl_result.setWordWrap(True)

        self.btn_export = QPushButton("Export GeoTIFF (EPSG:4326)…")
        self.btn_export.clicked.connect(self._export_geotiff)
        self.btn_export.setEnabled(False)

        self.lbl_export_info = QLabel("Output: —")  # size & resolution info
        self.lbl_export_info.setWordWrap(True)

        v_out.addWidget(self.btn_compute)
        v_out.addLayout(opt_layout)
        # self.lbl_result hidden: not added to layout
        v_out.addWidget(self.btn_export)
        # self.lbl_export_info hidden: not added to layout
        right.addWidget(grp_out)
        right.addStretch()

    def _build_menu_toolbar(self):
        act_open = QAction("Open PDF…", self); act_open.triggered.connect(self._choose_pdf)
        act_raster = QAction("Rasterize", self); act_raster.triggered.connect(self._rasterize)
        act_export = QAction("Export GeoTIFF…", self); act_export.triggered.connect(self._export_geotiff)

        tb = self.addToolBar("Main")
        tb.addAction(act_open)
        tb.addAction(act_raster)
        tb.addAction(act_export)

    # ------------------- Populate SPCS -------------------
    def _populate_stateplane_combo(self):
        self.cmb_spcs.clear()
        if query_crs_info is None:
            LOG.warning("pyproj.database.query_crs_info not available. Update pyproj to list EPSG.")
            self.cmb_spcs.addItem("— (pyproj too old, cannot list)", userData=None)
            return
        try:
            infos = list(query_crs_info(auth_name="EPSG"))
        except Exception:
            LOG.exception("Failed to query EPSG database via pyproj.")
            self.cmb_spcs.addItem("— (EPSG query failed)", userData=None)
            return

        candidates = []
        for inf in infos:
            try:
                name = inf.name
                code = inf.code
            except Exception:
                continue
            if not name or not code:
                continue
            if not name.startswith("NAD83 /"):
                continue
            if "UTM zone" in name or "UTM Zone" in name:
                continue
            # Exclude variant realizations if you want NAD83 “classic”
            if ("NAD83(HARN)" in name) or ("NAD83(NSRS2007)" in name) or ("NAD83(2011)" in name):
                continue
            label = f"{name} [EPSG:{code}]"
            candidates.append((label, int(code)))

        candidates.sort(key=lambda t: t[0].lower())
        if not candidates:
            self.cmb_spcs.addItem("— (no NAD83 SPCS found)", userData=None)
            return

        for label, epsg in candidates:
            self.cmb_spcs.addItem(label, userData=epsg)

        self.cmb_spcs.setCurrentIndex(0)
        self._on_spcs_changed()
        self.cmb_spcs.currentIndexChanged.connect(self._on_spcs_changed)
        LOG.info("Loaded SPCS NAD83 options: %d", len(candidates))

    # ------------------- Mode / SPCS events -------------------
    def _on_mode_changed(self, idx):
        text = self.cmb_mode.currentText()
        if "Lat/Lon" in text:
            self.input_mode = "latlon"
            self.cmb_spcs.setEnabled(False)
            self.lbl_epsg.setText("EPSG: 4326 (WGS84)")
            self._set_table_headers_for_latlon()
        else:
            self.input_mode = "xy"
            self.cmb_spcs.setEnabled(True)
            self._on_spcs_changed()
            self._set_table_headers_for_xy()

    def _on_spcs_changed(self):
        epsg = self.cmb_spcs.currentData()
        if epsg is None:
            self.selected_epsg = None
            self.lbl_epsg.setText("EPSG: —")
            return
        self.selected_epsg = int(epsg)
        self.lbl_epsg.setText(f"EPSG: {self.selected_epsg}")
        LOG.info("Selected CRS: EPSG:%s", self.selected_epsg)

    # ------------------- Table headers -------------------
    def _set_table_headers_for_latlon(self):
        self.tbl.setColumnCount(4)
        self.tbl.setHorizontalHeaderLabels(["Col (x)", "Row (y)", "Lat", "Lon"])

    def _set_table_headers_for_xy(self):
        self.tbl.setColumnCount(4)
        self.tbl.setHorizontalHeaderLabels(["Col (x)", "Row (y)", "X (Easting)", "Y (Northing)"])

    # ------------------- PDF / Raster -------------------
    def _choose_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF (*.pdf)")
        if not path:
            return
        LOG.info("Selected PDF: %s", path)
        try:
            doc = fitz.open(path)
            self.page_count = len(doc)
            doc.close()
        except Exception:
            LOG.exception("Could not open PDF")
            QMessageBox.critical(self, "Error", "Could not open the PDF.")
            return

        self.pdf_path = path
        self.lbl_pdf.setText(f"PDF: {os.path.basename(path)}  ({self.page_count} pages)")
        self.spn_page.setMaximum(self.page_count)
        self.spn_page.setValue(1)
        self.view.scene.clear()
        self._clear_points()
        self.btn_export.setEnabled(False)
        self._rgb_array = None
        self._pil_img = None
        self._qimage = None
        self._pixmap = None
        self.img_w = 0
        self.img_h = 0
        self._last_pdf_for_raster = None
        self._last_page_idx_for_raster = None
        self._update_export_info()

    def _rasterize(self):
        if not self.pdf_path:
            QMessageBox.warning(self, "Attention", "Select a PDF first.")
            return

        page_idx = self.spn_page.value() - 1
        dpi = self.spn_dpi.value()
        max_mp = float(self.spn_maxmp.value())
        include_ann = self.chk_ann.isChecked()

        # Are we re-rasterizing same PDF/page?
        same_target = (
            self.pdf_path == self._last_pdf_for_raster and
            page_idx == self._last_page_idx_for_raster and
            self.img_w > 0 and self.img_h > 0
        )

        # Save current markers if we will preserve/reposition
        saved_positions = []
        saved_vals_col2_col3 = []
        if same_target and len(self.markers) > 0:
            for m in self.markers:
                p = m.pos()
                saved_positions.append((float(p.x()), float(p.y())))
            for r in range(self.tbl.rowCount()):
                v2 = self.tbl.item(r, 2).text().strip() if self.tbl.item(r, 2) else ""
                v3 = self.tbl.item(r, 3).text().strip() if self.tbl.item(r, 3) else ""
                saved_vals_col2_col3.append((v2, v3))
            LOG.info("Preserving %d points for relocation after re-rasterization.", len(saved_positions))

        try:
            rgb_arr, eff_dpi = pdf_page_to_array_safe(
                self.pdf_path, page_idx, dpi, max_mp, include_ann
            )
        except Exception:
            LOG.exception("Rasterization failed")
            QMessageBox.critical(self, "Error", "Rasterization failed.")
            return

        try:
            # Keep references alive; build QImage via PIL for display
            self._rgb_array = rgb_arr
            self._pil_img = Image.fromarray(self._rgb_array, mode='RGB')
            self._qimage = ImageQt(self._pil_img).copy()
            self._pixmap = QPixmap.fromImage(self._qimage)
            self.view.set_image(self._pixmap)

            new_w, new_h = self._pixmap.width(), self._pixmap.height()
            old_w, old_h = self.img_w, self.img_h

            self.img_w, self.img_h = new_w, new_h
            self._last_pdf_for_raster = self.pdf_path
            self._last_page_idx_for_raster = page_idx

            # Reset output state
            self.transform = None
            self.btn_export.setEnabled(False)
            self.lbl_result.setText("Result: —")

            if same_target and saved_positions:
                # Reposition proportionally
                sx = new_w / max(1.0, old_w)
                sy = new_h / max(1.0, old_h)
                LOG.info("Repositioning points with scale sx=%.5f, sy=%.5f", sx, sy)

                # Clear lists (scene cleared by set_image)
                self.markers = []
                self.tbl.setRowCount(0)

                for i, (x, y) in enumerate(saved_positions):
                    nx, ny = x * sx, y * sy
                    self._add_marker(nx, ny)
                    if i < len(saved_vals_col2_col3):
                        v2, v3 = saved_vals_col2_col3[i]
                        self.tbl.item(i, 2).setText(v2)
                        self.tbl.item(i, 3).setText(v3)
                LOG.info("Points relocated and values preserved.")
            else:
                self._clear_points()

            self.statusBar().showMessage(
                f"Page rasterized at {eff_dpi:.1f} DPI ({new_w}x{new_h}px)",
                6000
            )
            LOG.info("Image loaded into view (array)")
        except Exception:
            LOG.exception("Error showing the rasterized image")
            QMessageBox.critical(self, "Error", "Rasterized but could not show the image.")

        self._update_export_info()

    # ------------------- Markers interaction -------------------
    def _handle_mouse_move(self, x, y):
        self.statusBar().showMessage(f"col={x:.2f}, row={y:.2f}")

    def _scene_tolerance(self, px=14):
        s = abs(self.view.transform().m11())
        if s <= 1e-9:
            s = 1.0
        return px / s

    def _find_nearest_marker_index(self, x, y, tol_scene):
        best_i = -1
        best_d2 = float('inf')
        for i, m in enumerate(self.markers):
            pos = m.pos()
            dx = pos.x() - x
            dy = pos.y() - y
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_i >= 0 and best_d2 <= (tol_scene * tol_scene):
            return best_i
        return -1

    def _handle_left_click(self, x, y):
        self._add_marker(x, y)
        self._update_export_info()

    def _handle_right_click(self, x, y):
        tol = self._scene_tolerance(px=14)
        idx = self._find_nearest_marker_index(x, y, tol_scene=tol)
        if idx == -1:
            self.statusBar().showMessage("No nearby marker to delete.")
            return
        self._delete_marker(idx)
        self._update_export_info()

    def _add_marker(self, x, y):
        idx = len(self.markers)
        if idx >= 3:
            QMessageBox.information(self, "Limit", "You already have 3 points. Delete one to change.")
            return
        marker = MarkerItem(
            index=idx, x=x, y=y, img_w=self.img_w, img_h=self.img_h,
            move_callback=self._marker_moved, radius=6.0
        )
        self.view.scene.addItem(marker)
        self.markers.append(marker)

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        it_x = QTableWidgetItem(f"{x:.3f}"); it_x.setFlags(it_x.flags() & ~Qt.ItemFlag.ItemIsEditable)
        it_y = QTableWidgetItem(f"{y:.3f}"); it_y.setFlags(it_y.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tbl.setItem(r, 0, it_x)
        self.tbl.setItem(r, 1, it_y)
        self.tbl.setItem(r, 2, QTableWidgetItem(""))  # Lat / X
        self.tbl.setItem(r, 3, QTableWidgetItem(""))  # Lon / Y
        LOG.debug("Point #%d added at col=%.3f, row=%.3f (mode=%s)", idx+1, x, y, self.input_mode)

    def _delete_marker(self, idx):
        if not (0 <= idx < len(self.markers)):
            return
        vals = []
        for r in range(self.tbl.rowCount()):
            v2 = self.tbl.item(r, 2).text().strip() if self.tbl.item(r, 2) else ""
            v3 = self.tbl.item(r, 3).text().strip() if self.tbl.item(r, 3) else ""
            vals.append((v2, v3))
        try:
            self.view.scene.removeItem(self.markers[idx])
        except Exception:
            pass
        del self.markers[idx]
        self.tbl.setRowCount(0)
        for i, m in enumerate(self.markers):
            m.update_index(i)
            pos = m.pos()
            self.tbl.insertRow(i)
            it_x = QTableWidgetItem(f"{pos.x():.3f}"); it_x.setFlags(it_x.flags() & ~Qt.ItemFlag.ItemIsEditable)
            it_y = QTableWidgetItem(f"{pos.y():.3f}"); it_y.setFlags(it_y.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(i, 0, it_x)
            self.tbl.setItem(i, 1, it_y)
            src = i if i < idx else i+1
            v2, v3 = vals[src] if src < len(vals) else ("", "")
            self.tbl.setItem(i, 2, QTableWidgetItem(v2))
            self.tbl.setItem(i, 3, QTableWidgetItem(v3))

        # Invalidate transform/export
        self.transform = None
        self.btn_export.setEnabled(False)
        self.lbl_result.setText("Result: —")
        self.statusBar().showMessage(f"Marker deleted. Now {len(self.markers)} point(s) remain.")
        LOG.info("Marker deleted. Remaining: %d", len(self.markers))

    def _marker_moved(self, index, x, y):
        if 0 <= index < self.tbl.rowCount():
            self.tbl.item(index, 0).setText(f"{x:.3f}")
            self.tbl.item(index, 1).setText(f"{y:.3f}")
            self.statusBar().showMessage(f"Moved point #{index+1}: col={x:.2f}, row={y:.2f}")

    def _clear_points(self):
        self.tbl.setRowCount(0)
        for m in self.markers:
            try:
                self.view.scene.removeItem(m)
            except Exception:
                pass
        self.markers = []
        self.transform = None
        self.btn_export.setEnabled(False)
        self.lbl_result.setText("Result: —")

    # ------------------- Read GCPs -------------------
    def _read_gcps_latlon(self):
        n = self.tbl.rowCount()
        if n != 3:
            raise ValueError("Exactly 3 points are required.")
        colrow, lats, lons = [], [], []
        for i in range(3):
            pos = self.markers[i].pos()
            colrow.append((float(pos.x()), float(pos.y())))
            lat = float(self.tbl.item(i, 2).text().strip())
            lon = float(self.tbl.item(i, 3).text().strip())
            if not (-90 <= lat <= 90):
                raise ValueError(f"Row {i+1}: Lat out of range (-90 to 90)")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Row {i+1}: Lon out of range (-180 to 180)")
            lats.append(lat); lons.append(lon)
        return colrow, np.array(lons), np.array(lats)

    def _read_gcps_xy_then_to_latlon(self):
        if self.selected_epsg is None:
            raise ValueError("Select a NAD83 State Plane projection in the dropdown.")
        try:
            crs_src = CRS.from_epsg(self.selected_epsg)
            transformer = Transformer.from_crs(crs_src, CRS.from_epsg(4326), always_xy=True)
        except Exception as e:
            raise ValueError(f"Could not create transformer for EPSG:{self.selected_epsg}: {e}")

        n = self.tbl.rowCount()
        if n != 3:
            raise ValueError("Exactly 3 points are required.")
        colrow, lons, lats = [], [], []
        for i in range(3):
            pos = self.markers[i].pos()
            colrow.append((float(pos.x()), float(pos.y())))
            x_txt = self.tbl.item(i, 2).text().strip()
            y_txt = self.tbl.item(i, 3).text().strip()
            x = float(x_txt)  # Easting
            y = float(y_txt)  # Northing
            lon, lat = transformer.transform(x, y)  # (lon, lat)
            lons.append(lon); lats.append(lat)
        return colrow, np.array(lons), np.array(lats)

    # ------------------- Compute / Export -------------------
    def _compute_transform(self):
        if self._rgb_array is None:
            QMessageBox.warning(self, "Attention", "Rasterize a PDF page first.")
            return
        if len(self.markers) != 3:
            QMessageBox.warning(self, "Attention", "Add exactly 3 points and complete the table.")
            return
        try:
            if self.input_mode == "latlon":
                colrow, lons, lats = self._read_gcps_latlon()
            else:
                colrow, lons, lats = self._read_gcps_xy_then_to_latlon()

            transform, rms_lon, rms_lat, ang_deg = compute_affine_transform(colrow, lons, lats)
        except Exception:
            LOG.exception("Error computing transform")
            QMessageBox.critical(self, "Error", "Could not compute the transform.")
            return

        self.transform = transform
        self.btn_export.setEnabled(True)
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        txt = (
            "Result:\n"
            f"  Transform (Affine):\n"
            f"    lon = {a:.8f}*col + {b:.8f}*row + {c:.8f}\n"
            f"    lat = {d:.8f}*col + {e:.8f}*row + {f:.8f}\n"
            f"  RMS lon: {rms_lon:.8e}°, RMS lat: {rms_lat:.8e}°\n"
            f"  Approx rotation (image x-axis → geo): {ang_deg:.3f}°"
        )
        self.lbl_result.setText(txt)
       
        LOG.info("Affine transform ready. Rot≈ %.3f° | RMS: lon=%.2e°, lat=%.2e°",
                 ang_deg, rms_lon, rms_lat)

        # Update export info (geo resolution)
        self._update_export_info()

    def _export_geotiff(self):
        if self.transform is None:
            QMessageBox.warning(self, "Attention", "Compute the transform first.")
            return
        if self._rgb_array is None:
            QMessageBox.warning(self, "Attention", "No rasterized image available.")
            return

        base_sug = "georef.tif"
        if self.pdf_path:
            base = os.path.splitext(os.path.basename(self.pdf_path))[0]
            base_sug = f"{base}_p{self.spn_page.value()}_georef.tif"

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save GeoTIFF", base_sug, "GeoTIFF (*.tif *.tiff)"
        )
        if not out_path:
            return

        try:
            idx = self.cmb_codec.currentIndex()
            if idx == 0:
                codec = 'jpeg'
                grayscale = False
            elif idx == 1:
                codec = 'deflate'
                grayscale = True
            elif idx == 2:
                codec = 'deflate'
                grayscale = False
            else:
                codec = 'jpeg2000'
                grayscale = False

            write_geotiff_from_array(
                self._rgb_array, self.transform, out_path,
                crs_epsg=4326, codec=codec, grayscale=grayscale, jpeg_quality=100
            )
            QMessageBox.information(
                self, "Done",
                f"GeoTIFF created:\n{out_path}"
            )
        except Exception:
            LOG.exception("Error exporting GeoTIFF")
            QMessageBox.critical(self, "Error", "Could not write the GeoTIFF.")

    # ------------------- Export info -------------------
    def _estimate_tif_sizes_mb(self, w, h, bands=3, dtype_bytes=1):
        """Return (uncompressed, est_min, est_max) in MiB, LZW ~40–80%."""
        uncompressed = (w * h * bands * dtype_bytes) / (1024 * 1024)
        min_est = uncompressed * 0.40
        max_est = uncompressed * 0.80
        return uncompressed, min_est, max_est

    def _update_export_info(self):
        if self._pixmap is None:
            self.lbl_export_info.setText("Output: —")
            return
        w = self._pixmap.width()
        h = self._pixmap.height()
        bands = 3
        if self.cmb_codec.currentIndex() == 1:
            bands = 1
        uncmp, est_min, est_max = self._estimate_tif_sizes_mb(w, h, bands=bands)
        txt = (f"Output (image): {w}×{h} px, {bands} band(s), uint8.\n"
               f"Estimated GeoTIFF size: ~{est_min:.1f}–{est_max:.1f} MiB "
               f"(raw ≈ {uncmp:.1f} MiB).")

        if self.transform is not None:
            a, b, c, d, e, f = self.transform.a, self.transform.b, self.transform.c, self.transform.d, self.transform.e, self.transform.f
            res_x_deg = math.hypot(a, d)
            res_y_deg = math.hypot(b, e)
            col_c = w / 2.0
            row_c = h / 2.0
            lon_c = a * col_c + b * row_c + c
            lat_c = d * col_c + e * row_c + f
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_c))
            res_x_m = math.hypot(a * m_per_deg_lon, d * m_per_deg_lat)
            res_y_m = math.hypot(b * m_per_deg_lon, e * m_per_deg_lat)
            txt += (f"\nApprox geospatial resolution: "
                    f"{res_x_deg:.6f}°/px × {res_y_deg:.6f}°/px  |  "
                    f"{res_x_m:.2f} m/px × {res_y_m:.2f} m/px (lat≈{lat_c:.5f}°).")
        self.lbl_export_info.setText(txt)

# --------------------------- main ------------------------------------
def main():
    app = QApplication(sys.argv)
    w = GeoRefApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
