from tests import utils
import pytest
import tempfile
import os

# Qt imports
from PySide6 import QtCore, QtWidgets, QtGui

from tests import utils


from gpanel256.gui.plugins.fields_editor import widgets
from gpanel256.core import sql
from gpanel256.config import Config


@pytest.fixture
def conn():
    return utils.create_conn()


def test_plugin(conn, qtbot):
    plugin = widgets.FieldsEditorWidget()
    plugin.mainwindow = utils.create_mainwindow()
    plugin.on_open_project(conn)
