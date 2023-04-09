import os
import yaml
import typing
from PySide6.QtCore import QStandardPaths, QDir, QFile, QFileInfo


class Config:


    USER_CONFIG_PATH = QDir(
        QStandardPaths.writableLocation(QStandardPaths.ConfigLocation)
        + QDir.separator()
        + "gpanel256"
    ).absoluteFilePath("config.yml")

    def __init__(self, section="app", config_path=None):
        self.section = section
        self.config_path = config_path or Config.user_config_path()

        self._user_config = dict()

        self.load()

    @classmethod
    def default_config_path(cls) -> str:

        return os.path.dirname(__file__) + os.path.sep + "default_config.yml"

    @classmethod
    def user_config_path(cls) -> str:

        return Config.USER_CONFIG_PATH

    def get(self, key: str, default=None) -> typing.Any:
        if self.section not in self._user_config:
            return default

        return self._user_config[self.section].get(key, default)

    def set(self, key: str, value: typing.Any):
        if self.section not in self._user_config:
            self._user_config[self.section] = {}

        self._user_config[self.section][key] = value

    def load(self):
        if not os.path.exists(self.config_path):
            self.load_from_path(self.default_config_path())
            self.save()

        self.load_from_path(self.config_path)

    def load_from_path(self, config_path):
        with open(config_path, "r") as stream:
            try:
                self._user_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                LOGGER.critical(exc)
            except KeyError as err:
                print(f"Could not read section {self.section} from config ")

    def save(self):
        if not os.path.exists(os.path.dirname(self.config_path)):
            try:
                os.makedirs(os.path.dirname(self.config_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        try:
            with open(self.config_path, "w") as stream:
                yaml.dump(self._user_config, stream)
        except IOError:
            LOGGER.warning(f"Could not write config file {self.config_path}")

    def reset(self):
        self.load_from_path(self.default_config_path())
        self.save()

    def __getitem__(self, key: str):
        if self.section in self._user_config:
            if self.section in self._user_config:
                return self._user_config[self.section][key]
            else:
                return None

    def __setitem__(self, key: str, value: typing.Any):
        self.set(key, value)

    def __contains__(self, key):
        return self.section in self._user_config and key in self._user_config[self.section]


if __name__ == "__main__":

    config = Config("global")
    config.load()
