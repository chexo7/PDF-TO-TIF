# -*- coding: utf-8 -*-
"""
GeoRef PDF → GeoTIFF (PyQt6)
• LOG detallado (terminal, panel y archivo temporal)
• Rasterización robusta (DPI + límite MP + markups)
• Zoom rueda, pan, cursor cruz
• Marcadores visibles, arrastrables (3 máx)
• Clic izq = añadir punto, clic der = borrar el más cercano
• GCPs en 2 modos:
    - Lat/Lon (EPSG:4326)
    - X/Y (State Plane NAD83) con dropdown de EPSG y conversión automática a Lon/Lat
• Afín por triangulación (rotación/escala incluidas) → GeoTIFF (EPSG:4326)
• NUEVO:
    - Re-rasterizar mismo PDF/página conserva puntos y datos (reescala sus posiciones)
    - Debajo de “Exportar”: resolución y tamaño estimado del GeoTIFF;
      tras “Calcular”, también resolución geoespacial (°/px y m/px)
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
    QGroupBox, QFormLayout, QStatusBar, QTextEdit, QComboBox,
    QGraphicsEllipseItem, QGraphicsSimpleTextItem
)

# --- pyproj para CRS y transformaciones ---
from pyproj import CRS, Transformer
try:
    from pyproj.database import query_crs_info
except Exception:
    query_crs_info = None

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

# Excepthook: captura excepciones no manejadas SIN cerrar la app
_def_excepthook = sys.excepthook
def _log_excepthook(exc_type, exc_value, exc_tb):
    """Captura excepciones no manejadas SIN cerrar la app."""
    try:
        tb_txt = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        LOG.critical("Excepción no capturada:\n%s", tb_txt)
        try:
            app = QApplication.instance()
            if app is not None:
                msg = "Ocurrió un error inesperado. Revisa el LOG.\n\n" + tb_txt[:1500]
                QMessageBox.critical(None, "Error no capturado", msg)
        except Exception:
            pass
    except Exception:
        pass
    # No llamar al excepthook por defecto (evita terminar el proceso).
sys.excepthook = _log_excepthook
LOG.info("Log file: %s", _LOG_FILE)

# --------------------------- Rasterización ----------------------------
class RasterizeError(Exception):
    pass

def pdf_pagina_a_jpg_safe(pdf_path, page_index, dpi, max_megapixels=120.0, include_annots=True):
    """
    Rasteriza de forma robusta una página del PDF a JPG.
    Limita el tamaño final a `max_megapixels` (downscale manteniendo proporción).
    Devuelve (ruta_jpg, dpi_efectivo, (ancho_px, alto_px)).
    """
    LOG.info("Rasterizando PDF: %s | página: %s | dpi: %s | maxMP: %.1f | annots: %s",
             pdf_path, page_index + 1, dpi, max_megapixels, include_annots)

    if not os.path.isfile(pdf_path):
        raise RasterizeError(f"No se encontró el archivo PDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RasterizeError(f"No se pudo abrir el PDF: {e}")

    try:
        if not (0 <= page_index < len(doc)):
            raise RasterizeError(f"Índice de página inválido. El PDF tiene {len(doc)} páginas.")
        page = doc.load_page(page_index)
        rect = page.rect  # en puntos (1/72 in)
        LOG.debug("Dimensiones PDF (pt): width=%.2f, height=%.2f", rect.width, rect.height)

        # Escala base por DPI
        scale = max(dpi, 72) / 72.0
        w_px = rect.width * scale
        h_px = rect.height * scale
        mp = (w_px * h_px) / 1e6
        LOG.debug("Tamaño estimado: %.1f x %.1f px (%.1f MP)", w_px, h_px, mp)

        # Cap de megapíxeles → downscale si excede
        eff_scale = scale
        if mp > max_megapixels:
            factor = math.sqrt(max_megapixels / mp)
            eff_scale = scale * factor
            LOG.warning("Downscale automático: factor=%.3f (para no exceder %.1f MP)", factor, max_megapixels)

        mat = fitz.Matrix(eff_scale, eff_scale)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False, annots=include_annots)
        except TypeError:
            # Compatibilidad con versiones sin 'annots'
            pix = page.get_pixmap(matrix=mat, alpha=False)
        w, h = pix.width, pix.height
        eff_dpi = 72.0 * eff_scale

        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_jpg = os.path.join(tempfile.gettempdir(), f"{base}_p{page_index+1}_{int(eff_dpi)}dpi.jpg")
        pix.save(out_jpg)
        LOG.info("Raster OK → %s (%dx%d px, dpi efectivo=%.1f)", out_jpg, w, h, eff_dpi)
        return out_jpg, eff_dpi, (w, h)
    except Exception as e:
        tb = traceback.format_exc()
        raise RasterizeError(f"Falló la rasterización: {e}\nTraceback:\n{tb}")
    finally:
        doc.close()

# --------------------------- Geo-transform ---------------------------
def calcular_transformacion_afin(colrow_pts, lons, lats):
    """
    Calcula transformación afín exacta con 3 GCPs: (col,row) → (lon,lat).
      lon = a*col + b*row + c
      lat = d*col + e*row + f
    Devuelve: (Affine, rms_lon, rms_lat, angulo_deg)
    """
    cols = np.array([p[0] for p in colrow_pts], dtype=float)
    rows = np.array([p[1] for p in colrow_pts], dtype=float)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    if len(cols) != 3:
        raise ValueError("Se requieren exactamente 3 puntos para la triangulación.")

    M = np.column_stack([cols, rows, np.ones_like(cols)])  # (N x 3)

    params_lon = np.linalg.solve(M, lons)
    params_lat = np.linalg.solve(M, lats)
    a, b, c = params_lon
    d, e, f = params_lat

    lons_pred = M @ params_lon
    lats_pred = M @ params_lat
    rms_lon = float(np.sqrt(np.mean((lons_pred - lons) ** 2)))
    rms_lat = float(np.sqrt(np.mean((lats_pred - lats) ** 2)))

    ang_rad = math.atan2(d, a)  # dirección del eje x-imagen en (lon,lat)
    ang_deg = math.degrees(ang_rad)

    transform = Affine(a, b, c, d, e, f)
    return transform, rms_lon, rms_lat, ang_deg

def escribir_geotiff_desde_jpg(jpg_path, transform, out_tif_path, crs_epsg=4326):
    """Escribe GeoTIFF con la transform y CRS dados."""
    img = Image.open(jpg_path).convert('RGB')
    arr = np.array(img)  # (H, W, 3)
    height, width = arr.shape[0], arr.shape[1]
    data = np.transpose(arr, (2, 0, 1))  # (3, H, W)

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 3,
        'dtype': data.dtype,
        'transform': transform,
        'crs': f"EPSG:{crs_epsg}",
        'compress': 'lzw'
    }

    with rasterio.open(out_tif_path, 'w', **profile) as dst:
        dst.write(data)

    LOG.info("GeoTIFF exportado → %s (EPSG:%s)", out_tif_path, crs_epsg)
    return out_tif_path

# --------------------------- Items de marcador -----------------------
class MarkerItem(QGraphicsEllipseItem):
    """Marcador circular movible, tamaño constante en pantalla, con etiqueta."""
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
        self.setToolTip(f"Punto #{index+1} (arrástrame)")

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
        self.setToolTip(f"Punto #{new_index+1} (arrástrame)")

# --------------------------- Vista de imagen -------------------------
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

        self.left_click_callback = None     # agregar punto
        self.right_click_callback = None    # borrar punto
        self.move_callback = None           # mostrar coords

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

# --------------------------- Ventana principal -----------------------
class QtLogHandler(logging.Handler):
    """Handler para enviar logs al QTextEdit de la UI."""
    def __init__(self, widget: QTextEdit):
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.widget.append(msg)
        except Exception:
            pass

class GeoRefApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoRef PDF → GeoTIFF (3 GCPs, Rot/Scale) + LOG + Zoom/Drag + SPCS NAD83")
        self.resize(1360, 930)

        # Estado
        self.pdf_path = None
        self.jpg_path = None
        self.page_count = 0
        self.transform = None

        # Evitar GC de imágenes (referencias vivas)
        self._pil_img = None
        self._qimage = None
        self._pixmap = None

        # Marcadores
        self.markers = []  # lista de MarkerItem
        self.img_w = 0
        self.img_h = 0

        # CRS seleccionados / modo
        self.input_mode = "latlon"  # "latlon" | "xy"
        self.selected_epsg = None   # EPSG del dropdown cuando input_mode == "xy"

        # Para preservar puntos al re-rasterizar mismo PDF/página
        self._last_pdf_for_raster = None
        self._last_page_idx_for_raster = None
        self._last_img_w = 0
        self._last_img_h = 0

        # UI
        self._build_ui()
        self._build_menu_toolbar()
        self.setStatusBar(QStatusBar())

        # Conectar LOG a panel
        self.qt_log_handler = QtLogHandler(self.txt_log)
        self.qt_log_handler.setLevel(logging.DEBUG)
        LOG.addHandler(self.qt_log_handler)

        # Callbacks del visor
        self.view.left_click_callback = self._handle_left_click
        self.view.right_click_callback = self._handle_right_click
        self.view.move_callback = self._handle_mouse_move

        # Poblar SPCS NAD83
        self._populate_stateplane_combo()

        LOG.info("==== GeoRef GUI iniciada ====")

    # ------------------- UI -------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Panel izquierdo: imagen
        self.view = ImageView()
        main.addWidget(self.view, stretch=3)

        # Panel derecho: controles
        right = QVBoxLayout()
        main.addLayout(right, stretch=2)

        # Grupo: PDF → JPG
        grp_pdf = QGroupBox("PDF → JPG")
        f1 = QFormLayout(grp_pdf)

        self.lbl_pdf = QLabel("PDF: (no cargado)")
        self.btn_pdf = QPushButton("Seleccionar PDF…")
        self.btn_pdf.clicked.connect(self._choose_pdf)

        # Parámetros raster
        self.spn_page = QSpinBox(); self.spn_page.setMinimum(1); self.spn_page.setMaximum(1); self.spn_page.setValue(1)
        self.spn_dpi = QSpinBox(); self.spn_dpi.setRange(72, 1200); self.spn_dpi.setValue(300)
        self.spn_maxmp = QSpinBox(); self.spn_maxmp.setRange(10, 600); self.spn_maxmp.setValue(120)  # en MP
        self.chk_ann = QCheckBox("Incluir anotaciones/markups"); self.chk_ann.setChecked(True)

        self.btn_raster = QPushButton("Rasterizar página → JPG")
        self.btn_raster.clicked.connect(self._rasterize)

        f1.addRow(self.lbl_pdf, self.btn_pdf)
        f1.addRow(QLabel("Página"), self.spn_page)
        f1.addRow(QLabel("DPI"), self.spn_dpi)
        f1.addRow(QLabel("Máx. Megapíxeles"), self.spn_maxmp)
        f1.addRow(self.chk_ann)
        f1.addRow(self.btn_raster)
        right.addWidget(grp_pdf)

        # Grupo: Modo de GCPs y Proyección
        grp_mode = QGroupBox("Coordenadas de GCPs (entrada)")
        fmode = QFormLayout(grp_mode)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Lat/Lon (EPSG:4326)", "X/Y (State Plane NAD83)"])
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.cmb_spcs = QComboBox()
        self.cmb_spcs.setEditable(True)  # permite escribir para filtrar
        self.cmb_spcs.setEnabled(False)  # solo en modo X/Y
        self.lbl_epsg = QLabel("EPSG: —")

        fmode.addRow(QLabel("Modo"), self.cmb_mode)
        fmode.addRow(QLabel("SPCS NAD83"), self.cmb_spcs)
        fmode.addRow(QLabel("EPSG detectado"), self.lbl_epsg)
        right.addWidget(grp_mode)

        # Grupo: GCPs (3 puntos)
        grp_gcps = QGroupBox("GCPs: 3 puntos (Triangulación)")
        v_gcps = QVBoxLayout(grp_gcps)

        self.tbl = QTableWidget(0, 4)
        self._set_table_headers_for_latlon()
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(self.tbl.EditTrigger.AllEditTriggers)

        hint = QLabel("Click izq: añade punto (máx. 3). Click der: borra punto cercano. "
                      "Arrastra los marcadores para ajustar.")
        self.btn_clear = QPushButton("Borrar todos")
        self.btn_clear.clicked.connect(self._clear_points)

        v_gcps.addWidget(hint)
        v_gcps.addWidget(self.tbl)
        v_gcps.addWidget(self.btn_clear)
        right.addWidget(grp_gcps)

        # Grupo: Transformación y salida
        grp_out = QGroupBox("Transformación y salida")
        v_out = QVBoxLayout(grp_out)

        self.btn_compute = QPushButton("Calcular transformación (rotación y escala incluidas)")
        self.btn_compute.clicked.connect(self._compute_transform)

        self.lbl_result = QLabel("Resultado: —")
        self.lbl_result.setWordWrap(True)

        self.btn_export = QPushButton("Exportar GeoTIFF EPSG:4326…")
        self.btn_export.clicked.connect(self._export_geotiff)
        self.btn_export.setEnabled(False)

        self.lbl_export_info = QLabel("Salida: —")  # resolución y tamaño estimado
        self.lbl_export_info.setWordWrap(True)

        v_out.addWidget(self.btn_compute)
        v_out.addWidget(self.lbl_result)
        v_out.addWidget(self.btn_export)
        v_out.addWidget(self.lbl_export_info)
        right.addWidget(grp_out)

        # Grupo: LOG
        grp_log = QGroupBox("LOG (también en archivo y consola)")
        v_log = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        v_log.addWidget(self.txt_log)
        right.addWidget(grp_log)

        right.addStretch()

    def _build_menu_toolbar(self):
        act_open = QAction("Abrir PDF…", self); act_open.triggered.connect(self._choose_pdf)
        act_raster = QAction("Rasterizar", self); act_raster.triggered.connect(self._rasterize)
        act_export = QAction("Exportar GeoTIFF…", self); act_export.triggered.connect(self._export_geotiff)

        tb = self.addToolBar("Main")
        tb.addAction(act_open)
        tb.addAction(act_raster)
        tb.addAction(act_export)

    # ------------------- Populate SPCS -------------------
    def _populate_stateplane_combo(self):
        self.cmb_spcs.clear()
        if query_crs_info is None:
            LOG.warning("pyproj.database.query_crs_info no disponible. Actualiza pyproj para listar EPSG.")
            self.cmb_spcs.addItem("— (pyproj muy antiguo, no se pudo listar)", userData=None)
            return
        try:
            infos = list(query_crs_info(auth_name="EPSG"))
        except Exception:
            LOG.exception("No se pudo consultar la base EPSG con pyproj.")
            self.cmb_spcs.addItem("— (falló consulta a EPSG)", userData=None)
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
            # Excluir variantes si quieres solo NAD83 “clásico”
            if "NAD83(HARN)" in name or "NAD83(NSRS2007)" in name or "NAD83(2011)" in name:
                continue
            label = f"{name} [EPSG:{code}]"
            candidates.append((label, int(code)))

        candidates.sort(key=lambda t: t[0].lower())
        if not candidates:
            self.cmb_spcs.addItem("— (no se encontraron SPCS NAD83)", userData=None)
            return

        for label, epsg in candidates:
            self.cmb_spcs.addItem(label, userData=epsg)

        self.cmb_spcs.setCurrentIndex(0)
        self._on_spcs_changed()
        self.cmb_spcs.currentIndexChanged.connect(self._on_spcs_changed)
        LOG.info("SPCS NAD83 cargadas: %d opciones", len(candidates))

    # ------------------- Eventos de modo / SPCS -------------------
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
        LOG.info("CRS seleccionado: EPSG:%s", self.selected_epsg)

    # ------------------- Utilidades tabla -------------------
    def _set_table_headers_for_latlon(self):
        self.tbl.setColumnCount(4)
        self.tbl.setHorizontalHeaderLabels(["Col (x)", "Row (y)", "Lat", "Lon"])

    def _set_table_headers_for_xy(self):
        self.tbl.setColumnCount(4)
        self.tbl.setHorizontalHeaderLabels(["Col (x)", "Row (y)", "X (Easting)", "Y (Northing)"])

    # ------------------- PDF / Raster -------------------
    def _choose_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar PDF", "", "PDF (*.pdf)")
        if not path:
            return
        LOG.info("PDF seleccionado: %s", path)
        try:
            doc = fitz.open(path)
            self.page_count = len(doc)
            doc.close()
        except Exception:
            LOG.exception("No se pudo abrir el PDF")
            QMessageBox.critical(self, "Error", "No se pudo abrir el PDF. Revisa el LOG.")
            return

        self.pdf_path = path
        self.lbl_pdf.setText(f"PDF: {os.path.basename(path)}  ({self.page_count} páginas)")
        self.spn_page.setMaximum(self.page_count)
        self.spn_page.setValue(1)
        self.jpg_path = None
        self.view.scene.clear()
        self._clear_points()
        self.btn_export.setEnabled(False)
        self._last_pdf_for_raster = None
        self._last_page_idx_for_raster = None
        self._last_img_w = 0
        self._last_img_h = 0
        self._update_export_info()

    def _rasterize(self):
        if not self.pdf_path:
            QMessageBox.warning(self, "Atención", "Primero selecciona un PDF.")
            return

        page_idx = self.spn_page.value() - 1
        dpi = self.spn_dpi.value()
        max_mp = float(self.spn_maxmp.value())
        include_ann = self.chk_ann.isChecked()

        # ¿Vamos a re-rasterizar el mismo PDF/página?
        same_target = (
            self.pdf_path == self._last_pdf_for_raster and
            page_idx == self._last_page_idx_for_raster and
            self.img_w > 0 and self.img_h > 0
        )

        # Guardar puntos/valores actuales si se preservan
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
            LOG.info("Preservando %d puntos para reubicación tras re-rasterizar.", len(saved_positions))

        try:
            jpg_path, eff_dpi, (w, h) = pdf_pagina_a_jpg_safe(self.pdf_path, page_idx, dpi, max_mp, include_ann)
            self.jpg_path = jpg_path
        except Exception:
            LOG.exception("Rasterización fallida")
            QMessageBox.critical(self, "Error", "La rasterización falló. Revisa el LOG para detalles.")
            return

        try:
            # Mantener referencias vivas; usar copy() para que QImage no comparta memoria con PIL
            self._pil_img = Image.open(self.jpg_path).convert('RGB')
            self._qimage = ImageQt(self._pil_img).copy()
            self._pixmap = QPixmap.fromImage(self._qimage)
            self.view.set_image(self._pixmap)

            new_w, new_h = self._pixmap.width(), self._pixmap.height()
            old_w, old_h = self.img_w, self.img_h

            self.img_w, self.img_h = new_w, new_h
            self.jpg_path = jpg_path
            self._last_pdf_for_raster = self.pdf_path
            self._last_page_idx_for_raster = page_idx
            self._last_img_w = new_w
            self._last_img_h = new_h

            # Reset de estado de salida (hay nueva imagen)
            self.transform = None
            self.btn_export.setEnabled(False)
            self.lbl_result.setText("Resultado: —")

            if same_target and saved_positions:
                # Reposicionar proporcionalmente
                sx = new_w / max(1.0, old_w)
                sy = new_h / max(1.0, old_h)
                LOG.info("Recolocando puntos con escala sx=%.5f, sy=%.5f", sx, sy)

                # Limpiar listas (la escena ya fue limpiada por set_image)
                self.markers = []
                self.tbl.setRowCount(0)

                for i, (x, y) in enumerate(saved_positions):
                    nx, ny = x * sx, y * sy
                    self._add_marker(nx, ny)
                    # Restaurar columnas 2-3 (Lat/Lon o X/Y) si existen
                    if i < len(saved_vals_col2_col3):
                        v2, v3 = saved_vals_col2_col3[i]
                        self.tbl.item(i, 2).setText(v2)
                        self.tbl.item(i, 3).setText(v3)
                LOG.info("Puntos reubicados y datos preservados.")
            else:
                # Cambio de PDF/página o no había puntos → limpiar
                self._clear_points()

            self.statusBar().showMessage(
                f"Página rasterizada a {eff_dpi:.1f} DPI → {self.jpg_path} ({new_w}x{new_h}px)",
                6000
            )
            LOG.info("Imagen cargada en la vista: %s", self.jpg_path)
        except Exception:
            LOG.exception("Error mostrando la imagen rasterizada")
            QMessageBox.critical(self, "Error", "Se rasterizó pero no se pudo mostrar la imagen. Revisa el LOG.")

        # Actualizar info de export (resolución y tamaño estimado, sin geoespacial aún)
        self._update_export_info()

    # ------------------- Interacción con puntos -------------------
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
        self._update_export_info()  # por si habilitamos después

    def _handle_right_click(self, x, y):
        tol = self._scene_tolerance(px=14)
        idx = self._find_nearest_marker_index(x, y, tol_scene=tol)
        if idx == -1:
            self.statusBar().showMessage("No hay marcador cercano para borrar.")
            return
        self._delete_marker(idx)
        self._update_export_info()

    def _add_marker(self, x, y):
        idx = len(self.markers)
        if idx >= 3:
            QMessageBox.information(self, "Límite", "Ya tienes 3 puntos. Borra alguno si deseas cambiar.")
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
        self.tbl.setItem(r, 2, QTableWidgetItem(""))  # Lat/X
        self.tbl.setItem(r, 3, QTableWidgetItem(""))  # Lon/Y
        LOG.debug("Punto #%d añadido en col=%.3f, row=%.3f (modo=%s)", idx+1, x, y, self.input_mode)

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

        # Invalida transform/export
        self.transform = None
        self.btn_export.setEnabled(False)
        self.lbl_result.setText("Resultado: —")
        self.statusBar().showMessage(f"Marcador borrado. Ahora hay {len(self.markers)} punto(s).")
        LOG.info("Borrado marcador. Restantes: %d", len(self.markers))

    def _marker_moved(self, index, x, y):
        if 0 <= index < self.tbl.rowCount():
            self.tbl.item(index, 0).setText(f"{x:.3f}")
            self.tbl.item(index, 1).setText(f"{y:.3f}")
            self.statusBar().showMessage(f"Movido punto #{index+1}: col={x:.2f}, row={y:.2f}")

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
        self.lbl_result.setText("Resultado: —")

    # ------------------- Lectura GCPs -------------------
    def _read_gcps_latlon(self):
        n = self.tbl.rowCount()
        if n != 3:
            raise ValueError("Se requieren exactamente 3 puntos.")
        colrow, lats, lons = [], [], []
        for i in range(3):
            pos = self.markers[i].pos()
            colrow.append((float(pos.x()), float(pos.y())))
            lat = float(self.tbl.item(i, 2).text().strip())
            lon = float(self.tbl.item(i, 3).text().strip())
            if not (-90 <= lat <= 90):
                raise ValueError(f"Fila {i+1}: Lat fuera de rango (-90 a 90)")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Fila {i+1}: Lon fuera de rango (-180 a 180)")
            lats.append(lat); lons.append(lon)
        return colrow, np.array(lons), np.array(lats)

    def _read_gcps_xy_then_to_latlon(self):
        if self.selected_epsg is None:
            raise ValueError("Selecciona una proyección SPCS NAD83 en el desplegable.")
        try:
            crs_src = CRS.from_epsg(self.selected_epsg)
            transformer = Transformer.from_crs(crs_src, CRS.from_epsg(4326), always_xy=True)
        except Exception as e:
            raise ValueError(f"No se pudo crear el transformador para EPSG:{self.selected_epsg}: {e}")

        n = self.tbl.rowCount()
        if n != 3:
            raise ValueError("Se requieren exactamente 3 puntos.")
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
        if not self.jpg_path:
            QMessageBox.warning(self, "Atención", "Rasteriza primero una página del PDF.")
            return
        if len(self.markers) != 3:
            QMessageBox.warning(self, "Atención", "Marca exactamente 3 puntos y completa las columnas.")
            return
        try:
            if self.input_mode == "latlon":
                colrow, lons, lats = self._read_gcps_latlon()
            else:
                colrow, lons, lats = self._read_gcps_xy_then_to_latlon()

            transform, rms_lon, rms_lat, ang_deg = calcular_transformacion_afin(colrow, lons, lats)
        except Exception:
            LOG.exception("Error calculando la transformación")
            QMessageBox.critical(self, "Error", "No se pudo calcular la transformación. Revisa el LOG.")
            return

        self.transform = transform
        self.btn_export.setEnabled(True)
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        txt = (
            "Resultado:\n"
            f"  Transform (Affine):\n"
            f"    lon = {a:.8f}*col + {b:.8f}*row + {c:.8f}\n"
            f"    lat = {d:.8f}*col + {e:.8f}*row + {f:.8f}\n"
            f"  RMS lon: {rms_lon:.8e}°, RMS lat: {rms_lat:.8e}°\n"
            f"  Rotación aprox (eje x imagen → geo): {ang_deg:.3f}°"
        )
        self.lbl_result.setText(txt)
        LOG.info("Transformación afín lista. Rot≈ %.3f° | RMS: lon=%.2e°, lat=%.2e°", ang_deg, rms_lon, rms_lat)

        # Actualiza la info de export con resolución geoespacial
        self._update_export_info()

    def _export_geotiff(self):
        if self.transform is None:
            QMessageBox.warning(self, "Atención", "Calcula la transformación antes de exportar.")
            return
        if not self.jpg_path:
            QMessageBox.warning(self, "Atención", "No hay imagen rasterizada.")
            return

        base_sug = "georef.tif"
        if self.pdf_path:
            base = os.path.splitext(os.path.basename(self.pdf_path))[0]
            base_sug = f"{base}_p{self.spn_page.value()}_georef.tif"

        out_path, _ = QFileDialog.getSaveFileName(self, "Guardar GeoTIFF", base_sug, "GeoTIFF (*.tif *.tiff)")
        if not out_path:
            return

        try:
            escribir_geotiff_desde_jpg(self.jpg_path, self.transform, out_path, crs_epsg=4326)
            QMessageBox.information(self, "Listo", f"GeoTIFF creado:\n{out_path}\n\nLog: {_LOG_FILE}")
        except Exception:
            LOG.exception("Error exportando el GeoTIFF")
            QMessageBox.critical(self, "Error", "No se pudo escribir el GeoTIFF. Revisa el LOG.")

    # ------------------- Info de exportación -------------------
    def _estimate_tif_sizes_mb(self, w, h, bands=3, dtype_bytes=1):
        """Devuelve (mb_uncompressed, mb_min_est, mb_max_est) estimando LZW ~40–80%."""
        uncompressed = (w * h * bands * dtype_bytes) / (1024 * 1024)  # MiB
        min_est = uncompressed * 0.40
        max_est = uncompressed * 0.80
        return uncompressed, min_est, max_est

    def _update_export_info(self):
        if self._pixmap is None:
            self.lbl_export_info.setText("Salida: —")
            return
        w = self._pixmap.width()
        h = self._pixmap.height()
        uncmp, est_min, est_max = self._estimate_tif_sizes_mb(w, h)
        txt = f"Salida (imagen): {w}×{h} px, 3 bandas (uint8), LZW.\n" \
              f"Tamaño estimado GeoTIFF: ~{est_min:.1f}–{est_max:.1f} MiB (sin garantía; sin datos ≈ {uncmp:.1f} MiB)."

        # Si hay transform, agrega resolución geo aproximada
        if self.transform is not None:
            a, b, c, d, e, f = self.transform.a, self.transform.b, self.transform.c, self.transform.d, self.transform.e, self.transform.f
            # Vectores de un píxel en lon/lat:
            # Δcol=(1,0) → (a,d); Δrow=(0,1) → (b,e)
            res_x_deg = math.hypot(a, d)
            res_y_deg = math.hypot(b, e)
            # Lat del centro para convertir a metros/deg
            col_c = w / 2.0
            row_c = h / 2.0
            lon_c = a * col_c + b * row_c + c
            lat_c = d * col_c + e * row_c + f
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_c))
            # Longitud de los vectores en metros/píxel
            res_x_m = math.hypot(a * m_per_deg_lon, d * m_per_deg_lat)
            res_y_m = math.hypot(b * m_per_deg_lon, e * m_per_deg_lat)
            txt += f"\nResolución geoespacial aprox: " \
                   f"{res_x_deg:.6f}°/px × {res_y_deg:.6f}°/px  |  " \
                   f"{res_x_m:.2f} m/px × {res_y_m:.2f} m/px (lat≈{lat_c:.5f}°)."
        self.lbl_export_info.setText(txt)

# --------------------------- main ------------------------------------
def main():
    app = QApplication(sys.argv)
    w = GeoRefApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
