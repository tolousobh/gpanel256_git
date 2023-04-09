
from dataclasses import replace
import typing
from functools import cmp_to_key, partial
import functools
import time
import copy
import re
import sqlite3

# Qt imports
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


# Custom imports
from gpanel256.core import sql, command
from gpanel256.core.reader import BedReader
from gpanel256.gui import plugin, FIcon, style
from gpanel256.constants import DEFAULT_SELECTION_NAME
from gpanel256.config import Config
import gpanel256.commons as cm


from gpanel256.gui.widgets import (
    ChoiceButton,
    SampleDialog,
    SampleVariantDialog,
    PresetAction,
)

from gpanel256.gui import tooltip as toolTip
from gpanel256 import LOGGER
from gpanel256.gui.sql_thread import SqlThread

from gpanel256.gui import FormatterDelegate
from gpanel256.gui.formatters.cutestyle import CutestyleFormatter

import gpanel256.constants as cst

from PySide6.QtWidgets import *
import sys
from functools import partial


class GenotypeVerticalHeader(QHeaderView):


    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)

    def sizeHint(self):
        return QSize(30, super().sizeHint().height())

    def paintSection(self, painter: QPainter, rect: QRect, section: int):

        painter.setBrush(QBrush(QColor("red")))

        painter.save()
        super().paintSection(painter, rect, section)
        painter.restore()
        # default color
        default_color = "lightgray"

        number = self.model().get_genotype(section).get("classification")
        if number:
            classification = next(i for i in self.model().classifications if i["number"] == number)
            color = classification.get("color", default_color)
        else:
            color = default_color

        GENOTYPE_ICONS = {key: FIcon(val) for key, val in cst.GENOTYPE_ICONS.items()}
        genotype_sample_name = self.model().get_genotype(section)["name"]
        genotype_variant_id = self.model().get_genotype(section)["variant_id"]

        genotype = ""
        if genotype_variant_id and genotype_sample_name:
            genotype_infos = next(
                sql.get_genotypes(
                    self.model().conn,
                    genotype_variant_id,
                    ["gt"],
                    [genotype_sample_name],
                )
            )
            genotype = genotype_infos.get("gt", -1)

        if genotype == "NULL" or genotype is None or genotype == "":
            genotype_int = -1
        else:
            genotype_int = int(genotype)

        pix_icon = GENOTYPE_ICONS.get(genotype_int)  # , GENOTYPE_ICONS[-1])
        pix_icon.engine.setColor(color)

        # painter
        pen = QPen(QColor(color))
        pen.setWidth(6)
        painter.setPen(pen)
        painter.setBrush(QBrush(color))
        painter.drawLine(rect.left(), rect.top() + 1, rect.left(), rect.bottom() - 1)

        target = QRect(0, 0, 20, 20)
        pix = pix_icon.pixmap(target.size())
        target.moveCenter(rect.center() + QPoint(1, 1))

        painter.drawPixmap(target, pix)


class GenotypeModel(QAbstractTableModel):


    samples_are_loading = Signal(bool)
    error_raised = Signal(str)
    load_started = Signal()
    load_finished = Signal()
    interrupted = Signal()

    def __init__(self, conn: sqlite3.Connection = None, parent=None):
        super().__init__(parent)

        self.conn = conn

        self._genotypes = []

        self._fields = set()

        self._samples = set()

        # Current variant
        self._variant_id = 0

        self._headers = []
        self.fields_descriptions = {}

        self.classifications = []

        self._load_samples_thread = SqlThread(self.conn)

        self._load_samples_thread.started.connect(lambda: self.samples_are_loading.emit(True))
        self._load_samples_thread.finished.connect(lambda: self.samples_are_loading.emit(False))
        self._load_samples_thread.result_ready.connect(self.on_samples_loaded)
        self._load_samples_thread.error.connect(self.error_raised)

        self._user_has_interrupt = False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """override"""
        if parent == QModelIndex():
            return len(self._genotypes)
        else:
            return 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """override"""
        if parent == QModelIndex():
            return len(self._headers)
        else:
            return 0

    def get_genotype(self, row: int) -> dict:
        return self._genotypes[row]

    def get_samples(self) -> typing.List[str]:
        return self._samples

    def set_samples(self, samples: typing.List[str]):
        self._samples = list(set(samples))

    def get_fields(self) -> typing.List[str]:
        return self._fields

    def set_fields(self, fields: typing.List[str]):
        self._fields = list(set(fields))

    def set_variant_id(self, variant_id: int):
        self._variant_id = variant_id

    def get_variant_id(self) -> int:
        return self._variant_id

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> typing.Any:
        if not index.isValid():
            return None

        item = self._genotypes[index.row()]
        key = self._headers[index.column()]

        if role == Qt.DisplayRole:
            return item[key]

        if role == Qt.ToolTipRole:
            return self.get_tooltip(index.row())


    def headerData(self, section: int, orientation: Qt.Orientation, role: int):

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section < len(self._headers):
                return self._headers[section]

        # vertical header
        if role == Qt.ToolTipRole and orientation == Qt.Vertical:
            genotype = self.get_genotype(section)
            genotype_tooltip = toolTip.genotype_tooltip(data=genotype, conn=self.conn)
            return genotype_tooltip

        return None


    def load(self):
        if self.conn is None:
            return

        if not self.get_samples():
            self.clear()
            return

        if self.is_running():
            LOGGER.debug("Cannot load data. Thread is not finished. You can call interrupt() ")
            self.interrupt()



        used_fields = copy.deepcopy(self.get_fields()) or []
        if "classification" not in used_fields:
            used_fields.append("classification")

        load_samples_func = partial(
            sql.get_genotypes,
            variant_id=self.get_variant_id(),
            fields=used_fields,
            samples=self.get_samples(),
        )

        self._start_timer = time.perf_counter()



        self.load_started.emit()

        self._load_samples_thread.conn = self.conn
        self._load_samples_thread.start_function(lambda conn: list(load_samples_func(conn)))

    def sort(self, column: int, order: Qt.SortOrder) -> None:
        self.beginResetModel()

        sorting_key = self.headerData(column, Qt.Horizontal, Qt.DisplayRole)

        def field_sort(i1, i2):
            if i1[sorting_key] is None:
                return -1
            if i2[sorting_key] is None:
                return 1

            if i1[sorting_key] < i2[sorting_key]:
                return -1
            elif i1[sorting_key] == i2[sorting_key]:
                return 0
            else:
                return 1

        self._genotypes = sorted(
            self._genotypes,
            key=cmp_to_key(field_sort),
            reverse=order == Qt.DescendingOrder,
        )
        self.endResetModel()

    def interrupt(self):

        interrupted = False

        if self._load_samples_thread:
            if self._load_samples_thread.isRunning():
                self._user_has_interrupt = True
                self._load_samples_thread.interrupt()
                self._load_samples_thread.wait(1000)
                interrupted = True

        if interrupted:
            self.interrupted.emit()

    def is_running(self):
        if self._load_samples_thread:
            return self._load_samples_thread.isRunning()
        return False

    def edit(self, rows: list, data: dict):

        rows = sorted(rows, reverse=True)
        for row in rows:

            self._genotypes[row].update(data)

            new_data = copy.deepcopy(self._genotypes[row])
            del new_data["name"]

            sql.update_genotypes(self.conn, new_data)
            self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()))
            self.headerDataChanged.emit(Qt.Vertical, row, row)

    def clear(self):

        self.beginResetModel()
        self._genotypes.clear()
        self.endResetModel()
        self.load_finished.emit()





class GenotypesWidget(plugin.PluginWidget):

    ENABLE = True
    REFRESH_STATE_DATA = {"current_variant", "samples"}

    def __init__(self, parent=None, conn=None):
        super().__init__(parent)

        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(16, 16))
        self.delegate = FormatterDelegate()
        self.delegate.set_formatter(CutestyleFormatter())
        self.model = GenotypeModel()
        self.view = QTableView()
        self.view.setShowGrid(False)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSortingEnabled(True)
        self.view.setIconSize(QSize(16, 16))
        self.view.horizontalHeader().setHighlightSections(False)
        self.view.setModel(self.model)

        self.view.setVerticalHeader(GenotypeVerticalHeader())
        self.view.verticalHeader().setSectionsClickable(True)
        self.view.verticalHeader().sectionDoubleClicked.connect(
            self._on_double_clicked_vertical_header
        )

        self.view.setItemDelegate(self.delegate)

        self.add_sample_button = QPushButton(self.tr("Add samples ..."))
        self.add_sample_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        empty_widget = QFrame()
        empty_widget.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        empty_widget.setBackgroundRole(QPalette.Base)
        empty_widget.setAutoFillBackground(True)
        empty_layout = QVBoxLayout(empty_widget)
        empty_layout.setAlignment(Qt.AlignCenter)

        empty_layout.addWidget(QLabel("Add samples to display genotypes ..."))

        self.label = QLabel()
        self.label.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumHeight(30)

        font = QFont()
        font.setBold(True)
        self.label.setFont(font)

        self.error_label = QLabel()
        self.error_label.hide()
        self.error_label.setStyleSheet(
            "QWidget{{background-color:'{}'; color:'{}'}}".format("orange", "black")
        )

        self.setWindowIcon(FIcon(0xF0AA1))

        self.stack_layout = QStackedLayout()
        self.stack_layout.addWidget(empty_widget)
        self.stack_layout.addWidget(self.view)

        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.addWidget(self.toolbar)
        vlayout.addLayout(self.stack_layout)
        vlayout.addWidget(self.error_label)
        vlayout.addWidget(self.label)
        vlayout.setSpacing(0)
        self.setLayout(vlayout)

        self.view.doubleClicked.connect(self._on_double_clicked)
        self.model.error_raised.connect(self.show_error)
        self.model.load_finished.connect(self.on_load_finished)
        self.model.modelReset.connect(self.on_model_reset)
        self.setup_actions()

    def on_model_reset(self):
        if self.model.rowCount() > 0:
            self.stack_layout.setCurrentIndex(1)
            self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        else:
            self.stack_layout.setCurrentIndex(0)

        # self.view.horizontalHeader().setSectionResizeMode(QHeaderView.AdjustToContents)

    def setup_actions(self):



        self.fields_button = ChoiceButton()
        self.fields_button.prefix = "Fields"
        self.fields_button.empty_message = "gt"
        self.fields_button.setFixedWidth(100)
        self.fields_button.item_changed.connect(self.on_refresh)
        self.toolbar.addWidget(self.fields_button)

        self.toolbar.addAction(QIcon(), "Clear Fields", self.fields_button.uncheck_all)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        self.preset_menu = QMenu()
        self.preset_button = QPushButton()
        self.preset_button.setToolTip(self.tr("Presets"))
        self.preset_button.setIcon(FIcon(0xF035C))
        self.preset_button.setMenu(self.preset_menu)
        self.preset_button.setFlat(True)
        self.toolbar.addWidget(self.preset_button)

        self.load_presets()


    def _on_double_clicked(self):
        self._on_default_classification_changed()

    def _on_double_clicked_vertical_header(self):
        self._show_sample_variant_dialog()

    def contextMenuEvent(self, event: QContextMenuEvent):

        row = self.view.selectionModel().currentIndex().row()

        genotype = self.model.get_genotype(row)

        menu = QMenu(self)

        sample_name = genotype.get("name", "unknown")

        variant_id = genotype.get("variant_id", 0)
        variant_name = cm.find_variant_name(conn=self.conn, variant_id=variant_id, troncate=True)

        if genotype["sample_id"] and genotype["variant_id"]:

            if self.is_locked(genotype.get("sample_id", 0)):
                validation_menu_enable = False
                validation_menu_title = "Classification (locked)"
                tags_menu_title = "Tags (locked)"
            else:
                validation_menu_enable = True
                validation_menu_title = "Classification"
                tags_menu_title = "Tags"

            menu.addAction(
                FIcon(0xF064F),
                f"Edit Genotype '{sample_name}' - '{variant_name}'",
                self._show_sample_variant_dialog,
            )

            cat_menu = menu.addMenu(validation_menu_title)
            cat_menu.setEnabled(validation_menu_enable)

            for item in self.model.classifications:

                if genotype["classification"] == item["number"]:
                    icon = 0xF0133
                else:
                    icon = 0xF012F

                action = cat_menu.addAction(FIcon(icon, item["color"]), item["name"])
                action.setData(item["number"])
                action.triggered.connect(self._on_classification_changed)

            tags_menu = menu.addMenu(tags_menu_title)
            tags_menu.setEnabled(validation_menu_enable)

            tags_preset = Config("tags")

            for item in tags_preset.get("genotypes", []):

                icon = 0xF04F9

                action = tags_menu.addAction(FIcon(icon, item["color"]), item["name"])
                action.setData(item["name"])
                on_click = functools.partial(self._on_tags_changed, [item["name"]])
                action.triggered.connect(on_click)

            menu.exec_(event.globalPos())

    def is_locked(self, sample_id: int):
        config_classif = Config("classifications").get("samples", [])
        sample = sql.get_sample(self.conn, sample_id)
        sample_classif = sample.get("classification", None)

        if config_classif == None or sample_classif == None:
            return False

        locked = False
        for config in config_classif:
            if config["number"] == sample_classif and "lock" in config:
                if config["lock"] == True:
                    locked = True
        return locked

    def _show_sample_dialog(self):

        row = self.view.selectionModel().currentIndex().row()
        sample = self.model.get_genotype(row)
        if sample:

            dialog = SampleDialog(self.conn, sample["sample_id"])

            if dialog.exec_() == QDialog.Accepted:
                self.on_refresh()

    def _show_sample_variant_dialog(self):

        row = self.view.selectionModel().currentIndex().row()
        sample = self.model.get_genotype(row)

        sample_name = sample.get("name", "unknown")
        sample_id = sample.get("sample_id", None)
        variant_id = sample.get("variant_id", None)

        if sample and sample_id is not None and variant_id is not None:

            dialog = SampleVariantDialog(self.conn, sample["sample_id"], self.current_variant["id"])

            if dialog.exec_() == QDialog.Accepted:
                self.on_refresh()

        else:
            QMessageBox.information(
                self,
                "No genotype",
                self.tr(f"Sample '{sample_name}' does not have genotype for this variant"),
            )

    def _toggle_column(self, col: int, show: bool):
        if show:
            self.view.showColumn(col)
        else:
            self.view.hideColumn(col)


    def _on_clear_filters(self):

        self.on_refresh()

    def _create_filters(self, copy_existing_filters: bool = True) -> dict:

        indexes = self.view.selectionModel().selectedRows()
        if copy_existing_filters:
            filters = copy.deepcopy(self.mainwindow.get_state_data("filters"))
        else:
            filters = {}

        if not filters:
            root = "$or"
            filters["$or"] = []

        else:
            root = list(filters.keys())[0]
            filters[root] = [
                i for i in filters[root] if not list(i.keys())[0].startswith("samples")
            ]

        for index in indexes:
            sample_name = index.siblingAtColumn(0).data()
            if sample_name:
                key = f"samples.{sample_name}.gt"
                condition = {key: {"$gte": 1}}
                filters[root].append(condition)

        return filters

    def on_add_source(self):

        name, success = QInputDialog.getText(
            self, self.tr("Source Name"), self.tr("Get a source name ")
        )



        if success and name:

            sql.insert_selection_from_source(
                self.conn, name, "variants", self._create_filters(False)
            )

            if "source_editor" in self.mainwindow.plugins:
                self.mainwindow.refresh_plugin("source_editor")

        else:

            return




    def load_all_filters(self):
        self.load_fields()

    def load_samples(self):

        self.sample_selector.clear()
        for sample in sql.get_samples(self.conn):
            self.sample_selector.add_item(FIcon(0xF0B55), sample["name"], data=sample["name"])

    def load_fields(self):
        self.fields_button.blockSignals(True)
        self.fields_button.clear()
        for field in sql.get_field_by_category(self.conn, "samples"):
            self.fields_button.add_item(
                FIcon(0xF0835),
                field["name"],
                field["description"],
                data=field["name"],
            )
        self.fields_button.blockSignals(False)


    def show_error(self, message):
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def on_load_finished(self):
        self.show_error("")





if __name__ == "__main__":

    import sqlite3
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    conn = sqlite3.connect("/home/tolou/test3.db")
    conn.row_factory = sqlite3.Row

    view = GenotypesWidget()
    view.on_open_project(conn)
    view.show()

    app.exec()
