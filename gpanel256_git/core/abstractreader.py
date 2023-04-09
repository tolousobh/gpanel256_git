from abc import ABC, abstractmethod
import io
import gzip
import math
from collections import Counter

from urllib.parse import unquote

import gpanel256.constants as cst

from gpanel256 import LOGGER


class AbstractReader(ABC):


    def __init__(self, filename):
        self.filename = filename
        self.number_lines = None
        self.read_bytes = 0
        self.samples = list()

        self.file_size = 0

        self.ignored_fields = set()

    def get_variants(cls):

        raise NotImplementedError(cls.__class__.__name__)


    def get_fields(cls):

        raise NotImplementedError(cls.__class__.__name__)

    def get_samples(self) -> list:

        return []

    def get_metadatas(self) -> dict:

        return {}


    def progress(self) -> float:

        return -1


    def get_variants_count(self) -> int:

        return len(tuple(self.get_variants()))



