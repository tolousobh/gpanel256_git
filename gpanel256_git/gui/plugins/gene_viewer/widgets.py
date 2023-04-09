# Standard imports
import typing
import glob
import json
import os
import gzip
import sys
import sqlite3
import copy

# Qt imports
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtSql import *

# Custom imports
from gpanel256.gui import style, plugin, FIcon, MainWindow
from gpanel256.core.querybuilder import build_vql_query, build_sql_query

from gpanel256.core import sql
from gpanel256 import LOGGER

from gpanel256.gui.widgets import VqlSyntaxHighlighter
from gpanel256.config import Config





def overlap(interval1: list, interval2: list) -> typing.Tuple[bool, int, int]:
    _overlaps = True
    if interval2[0] <= interval1[0] and interval1[0] <= interval2[1]:
        start = interval1[0]
    elif interval1[0] <= interval2[0] and interval2[0] <= interval1[1]:
        start = interval2[0]
    else:
        _overlaps = False
        start, end = 0, 0

    if interval2[0] <= interval1[1] <= interval2[1]:
        end = interval1[1]
    elif interval1[0] <= interval2[1] <= interval1[1]:
        end = interval2[1]
    else:
        _overlaps = False
        start, end = 0, 0

    return (_overlaps, start, end)


class Gene:

    def __init__(self):
        self.cds_start = None
        self.cds_end = None
        self.exon_starts = None
        self.exon_ends = None
        self.tx_start = None
        self.tx_end = None
        self.transcript_name = None
        self.exon_count = 0
        self.variants = []

    def load(self, data: dict):
        """From sqlite dict"""
        self.cds_start = data["cds_start"]
        self.cds_end = data["cds_end"]
        self.exon_starts = [int(i) for i in data["exon_starts"].split(",")]
        self.exon_ends = [int(i) for i in data["exon_ends"].split(",")]
        self.tx_start = data["tx_start"]
        self.tx_end = data["tx_end"]
        self.transcript_name = data["transcript_name"]

        self.exon_count = len(self.exon_starts) if self.exon_starts else 0


class GeneView(QAbstractScrollArea):

    MOUSE_SELECT_MODE = 0  # Mouse clicking and dragging causes rectangle selection
    MOUSE_PAN_MODE = 1  # Mouse clicking and dragging causes view panning

    def __init__(self, parent=None):
        super().__init__(parent)

        self.variants = [(20761708, "red", 0.9), (20761808, "red", 0.2)]

        self.gene = None

        # style
        self.cds_height = 40
        self.exon_height = 30
        self.intron_height = 20

        # self.showMaximized()

        self.scale_factor = 1
        self.translation = 0

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.horizontalScrollBar().setRange(0, 0)
        self.horizontalScrollBar().valueChanged.connect(self.set_translation)

        self.resize(640, 200)

        self.region = None
        color = self.palette().color(QPalette.Highlight)
        self.region_pen = QPen(color)
        color.setAlphaF(0.3)
        self.region_brush = QBrush(color)

        self.viewport().setMouseTracking(True)

        self.set_mouse_mode(GeneView.MOUSE_SELECT_MODE)
        self.cursor_selects = False

        self.selected_exon = None
        self._sample_count = 1

    def set_gene(self, gene: Gene):

        self.gene = gene
        self.viewport().update()

    def get_mouse_mode(self) -> int:
        return self._mouse_mode

    def set_mouse_mode(self, mode: int):
        if mode == GeneView.MOUSE_SELECT_MODE:
            self._mouse_mode = mode
            QScroller.ungrabGesture(self.viewport())
            self.setCursor(Qt.ArrowCursor)
        elif mode == GeneView.MOUSE_PAN_MODE:
            self._mouse_mode = mode
            QScroller.grabGesture(self.viewport(), QScroller.LeftMouseButtonGesture)
            self.setCursor(Qt.OpenHandCursor)
        else:
            raise ValueError(
                "Cannot set mouse mode to %s (accepted modes are MOUSE_PAN_MODE,MOUSE_SELECT_MODE",
                str(mode),
            )

    @property
    def cursor_selects(self) -> bool:
        return self._cursor_selects

    @cursor_selects.setter
    def cursor_selects(self, value: bool):
        self._cursor_selects = value

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter()
        painter.begin(self.viewport())
        painter.setBrush(QBrush(self.palette().color(QPalette.Base)))

        self.area = self.area_rect()
        painter.setClipRect(self.area)

        if not self.gene:
            return

        if not (self.gene.tx_start and self.gene.tx_end):
            painter.drawText(
                self.area,
                Qt.AlignCenter,
                self.tr("No gene selected. Please chose one in the combobox"),
            )
            painter.end()
            return


        self._draw_variants(painter)


        self._draw_introns(painter)

        self._draw_exons(painter)

        self._draw_cds(painter)

        self._draw_region(painter)



        painter.end()


    def _draw_variants(self, painter: QPainter):
        if self.variants:
            painter.save()
            for variant in self.variants:

                pos = variant[0]
                selected = variant[1]
                af = variant[2]

                LOLIPOP_HEIGH = 100
                y = self.rect().center().y() - LOLIPOP_HEIGH * af - 10

                pos = self._dna_to_pixel(pos)
                pos = self._pixel_to_scroll(pos)

                col_line = self.palette().color(QPalette.Text)
                col_line.setAlphaF(0.6)
                painter.setPen(col_line)
                painter.drawLine(pos, self.rect().center().y(), pos, y)

                rect = QRect(0, 0, 10, 10)
                painter.setPen(self.palette().color(QPalette.Window))
                painter.setBrush(self.palette().color(QPalette.Highlight))

                col = self.palette().color(QPalette.Text)
                if selected:
                    col = QColor("red")

                col.setAlphaF(0.6)
                rect.moveCenter(QPoint(pos, y))
                painter.setBrush(QBrush(col))
                painter.drawEllipse(rect)

            painter.restore()












    @Slot(int)
    def set_scale(self, x: int):
        self.scale_factor = x

        min_scroll = 0
        max_scroll = (self.area_rect().width() * self.scale_factor) - self.area_rect().width()

        previous = self.horizontalScrollBar().value()
        previous_max = self.horizontalScrollBar().maximum()

        self.horizontalScrollBar().setRange(min_scroll, max_scroll)

        if previous_max > 1:
            new = previous * self.horizontalScrollBar().maximum() / previous_max
        else:
            new = self.horizontalScrollBar().maximum() / 2

        self.horizontalScrollBar().setValue(new)

    def set_translation(self, x: int):
        self.translation = x
        self.viewport().update()



    def keyPressEvent(self, event: QKeyEvent):

        super().keyPressEvent(event)


    def event(self, event: QEvent) -> bool:

        if event.type() == QEvent.Gesture:
            for g in event.gestures():
                if g.state() == Qt.GestureUpdated:
                    self.setCursor(Qt.ClosedHandCursor)
                else:
                    self.setCursor(Qt.OpenHandCursor)

        return super().event(event)

    def keyReleaseEvent(self, event: QKeyEvent):

        super().keyReleaseEvent(event)


    def mousePressEvent(self, event: QMouseEvent):
        if self.selected_exon != None:
            self.zoom_to_region(
                self.gene.exon_starts[self.selected_exon],
                self.gene.exon_ends[self.selected_exon],
            )

        if self.get_mouse_mode() == GeneView.MOUSE_SELECT_MODE:

            if event.button() == Qt.LeftButton:
                self.region = QRect(0, 0, 0, 0)
                self.region.setHeight(self.viewport().height())
                self.region.setLeft(event.position().x())
                self.region.setRight(event.position().x())

            if event.button() == Qt.RightButton:
                self.reset_zoom()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.region:
            self.region.setRight(event.position().x())


        self.viewport().update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.region:
            if self.region.normalized().width() < 5:
                self.region = None
                return

            self.region = self.region.normalized()

            if self.region.isEmpty():
                # There is no selection, dna_start - dna_end is too small, zooming will make no sense
                self.reset_zoom()
            else:
                dna_start = self._scroll_to_dna(self.region.left() - self.area_rect().left())

                dna_end = self._scroll_to_dna(self.region.right() - self.area_rect().left())

                print(self.region.width())
                self.zoom_to_region(dna_start, dna_end)
                self.region = None

        super().mouseReleaseEvent(event)

    def reset_zoom(self):
        if self.gene:
            self.zoom_to_region(self.gene.tx_start, self.gene.tx_end)
        else:
            self.set_scale(1)
            self.set_translation(0)

    def set_sample_count(self, value):
        self._sample_count = value

    def get_sample_count(self):
        return self._sample_count


class GeneViewerWidget(plugin.PluginWidget):

    ENABLE = True
    REFRESH_ONLY_VISIBLE = True
    REFRESH_STATE_DATA = {"current_variant"}

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(self.tr("Gene Viewer"))
        self.setWindowIcon(FIcon(0xF0684))

        self.view = GeneView()


        self.gene_name_combo = QComboBox()
        self.gene_name_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.gene_name_combo.setMaximumWidth(400)
        self.gene_name_combo.setEditable(True)
        self.gene_name_combo.lineEdit().setPlaceholderText("Gene name ...")

        self.transcript_name_combo = QComboBox()
        self.transcript_name_combo.setMaximumWidth(400)
        self.transcript_name_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.transcript_name_combo.setEditable(True)
        self.transcript_name_combo.lineEdit().setPlaceholderText("Transcript name ...")

        self.exon_combo = QComboBox()
        self.exon_combo.setMaximumWidth(400)

        self.toolbar = QToolBar()

        self.tool_widget = QWidget()
        hlayout = QHBoxLayout(self.tool_widget)
        hlayout.addWidget(self.gene_name_combo)
        hlayout.addWidget(self.transcript_name_combo)
        # hlayout.addWidget(self.exon_combo)
        hlayout.addStretch()
        self.toolbar.addWidget(self.tool_widget)

        self.empty_widget = QWidget()
        self.config_button = QLabel("Set a database from settings ... ")
        self.config_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        empty_layout = QVBoxLayout(self.empty_widget)
        empty_layout.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(self.config_button)

        self.stack_layout = QStackedLayout()
        self.stack_layout.addWidget(self.empty_widget)
        self.stack_layout.addWidget(self.view)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toolbar)
        main_layout.addLayout(self.stack_layout)

        self.exon_combo.activated.connect(
            lambda x: self.view.zoom_to_exon(self.exon_combo.currentData())
        )
        self.gene_name_combo.currentTextChanged.connect(self.on_selected_gene_changed)
        self.transcript_name_combo.currentTextChanged.connect(self.on_selected_transcript_changed)

        self.current_variant = {}

        self.selected_gene = ""
        self.selected_transcript = ""

        self.gene_names = []
        self.transcript_names = []

        self.gene_conn = None

    def on_open_project(self, conn):
        self.conn = conn
        try:
            self.load_config()
        except:
            LOGGER.debug("Cannot init gene viewer")

    def on_close_project(self):
        self.view.set_gene(None)


    def on_refresh(self):

        if not self.gene_conn:
            return

        """Called whenever this plugin needs updating."""
        self.current_variant = sql.get_variant(
            self.conn,
            self.mainwindow.get_state_data("current_variant")["id"],
            with_annotations=True,
        )

        # Config for gene_viewer
        config_gene_viewer = Config("gene_viewer")
        gene_field = config_gene_viewer.get("gene_field", "")
        transcript_field = config_gene_viewer.get("transcript_field", "")

        gene = ""
        if gene_field.split(".")[0] == "ann":
            if "annotations" in self.current_variant:
                if gene_field.split(".")[1] in self.current_variant["annotations"][0]:
                    gene = self.current_variant["annotations"][0][gene_field.split(".")[1]]
        else:
            if gene_field in self.current_variant:
                gene = self.current_variant[gene_field]

        transcript = ""
        # transcript from annotations
        if transcript_field.split(".")[0] == "ann":
            if "annotations" in self.current_variant:
                if transcript_field.split(".")[1] in self.current_variant["annotations"][0]:
                    transcript = self.current_variant["annotations"][0][
                        transcript_field.split(".")[1]
                    ].split(".")[0]
        else:
            if transcript_field in self.current_variant:
                transcript = self.current_variant[transcript_field].split(".")[0]

        self.transcript_name_combo.blockSignals(True)
        self.gene_name_combo.setCurrentText(gene)
        self.transcript_name_combo.blockSignals(False)
        self.transcript_name_combo.setCurrentText(transcript)
        self.update_view()


    def load_gene_names(self):

        if self.gene_conn:
            gene_names = [s["gene"] for s in self.gene_conn.execute("SELECT gene FROM genes")]
            self.gene_name_combo.clear()
            self.gene_name_combo.addItems(gene_names)

    def load_transcript_names(self):
        if self.gene_conn:
            transcript_names = (
                [
                    s["transcript_name"]
                    for s in self.gene_conn.execute(
                        f"SELECT transcript_name FROM genes WHERE gene = '{self.selected_gene}'"
                    )
                ]
                if self.selected_gene is not None
                else []
            )

            self.transcript_name_combo.clear()
            self.transcript_name_combo.addItems(transcript_names)
            if len(transcript_names) >= 1:
                # Select first transcript (by default)
                self.transcript_name_combo.setCurrentIndex(0)






    def update_view(self):

        if not self.current_variant:
            return

        gene = self.gene_name_combo.currentText()
        transcript = self.transcript_name_combo.currentText()
        query = f"SELECT transcript_name,tx_start,tx_end,cds_start,cds_end,exon_starts,exon_ends,gene FROM genes WHERE gene = '{gene}' AND transcript_name='{transcript}'"
        result = self.gene_conn.execute(query).fetchone()

        config_gene_viewer = Config("gene_viewer")
        gene_field = config_gene_viewer.get("gene_field", "")

        list_of_fields = []
        for field in sql.get_fields(self.conn):
            if field["category"] == "variants":
                name = field["name"]
                list_of_fields.append(name)
            if field["category"] == "annotations":
                name = field["name"]
                list_of_fields.append(f"ann.{name}")

        filters = {}
        if gene_field in list_of_fields:
            filters = {"$and": [{f"""{gene_field}""": gene}]}
        else:
            LOGGER.warning("Gene fields %s not in project", gene_field)

        fields = ["pos"]
        source = self.mainwindow.get_state_data("source")

        variants = []

        for v in sql.get_variants(self.conn, fields, source, filters, limit=None):
            pos = v["pos"]
            selected = pos == self.current_variant["pos"]
            variants.append((pos, selected, 0.5))

        if result is not None:
            gene = Gene()
            self.view.variants = variants
            gene.load(dict(result))
            self.view.set_gene(gene)





if __name__ == "__main__":

    pass

    import sys
    import sqlite3

    import os

    app = QApplication(sys.argv)

    conn = sqlite3.connect("/home/tolou/refGene.db", detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row

    gene_data = dict(conn.execute("SELECT * FROM genes WHERE gene = 'NOD2'").fetchone())

    print(gene_data)
    # print(gene_data)
    gene = Gene()
    gene.load(gene_data)

    print(gene.tx_start + 100)

    view = GeneView()
    view.set_gene(gene)
    view.variants = [(20766921, True, 0.5)]
    view.show()
    view.resize(600, 500)

    

    app.exec()
