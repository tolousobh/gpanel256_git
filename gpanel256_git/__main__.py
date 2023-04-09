import sys
from PySide6.QtCore import (
    QCoreApplication,
)
from PySide6.QtWidgets import QApplication, QSplashScreen, QColorDialog
from PySide6.QtGui import QColor
from gpanel256.config import Config
from gpanel256.gui import MainWindow, style
from gpanel256.gui.widgets import SplashScreen
import gpanel256.constants as cst
import os
import sqlite3
def main():

    app = QApplication(sys.argv)

    load_styles(app)

    splash = SplashScreen()

    splash.show()

    app.processEvents()


    w = MainWindow()
    splash.finish(w)
    w.show()

    app.exec()




def load_styles(app):

    config = Config("app")
    style_config = config.get("style", {})
    theme = style_config.get("theme", cst.BASIC_STYLE)

    mystyle = style.AppStyle()
    mystyle.load_theme(theme.lower() + ".yaml")
    app.setStyle(mystyle)

    for index, (key, color) in enumerate(mystyle.colors().items()):
        QColorDialog.setCustomColor(index, QColor(color))




def process_arguments(app):
    parser = QCommandLineParser()
    parser.addHelpOption()

    parser.addOption(show_version)


    config_option = QCommandLineOption(
        ["c", "config"],
        QCoreApplication.translate("config path", "Set the config path"),
        "config",
    )

    parser.addOption(config_option)


    parser.addOption(modify_verbosity)

    parser.process(app)

    if parser.isSet(show_version):
        print("gpanel256 " + __version__)
        exit()

    if parser.isSet(config_option):
        config_path = parser.value(config_option)
        if os.path.isfile(config_path):
            Config.USER_CONFIG_PATH = config_path

        else:
            LOGGER.error(f"{config_path} doesn't exists. Ignoring config")


    LOGGER.setLevel(parser.value(modify_verbosity).upper())


if __name__ == "__main__":
    main()
