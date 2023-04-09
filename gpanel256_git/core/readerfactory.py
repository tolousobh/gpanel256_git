from contextlib import contextmanager
import pathlib
import vcf

from gpanel256.core.reader import VcfReader, CsvReader
import gpanel256.commons as cm


from gpanel256 import LOGGER


def detect_vcf_annotation(filepath):
    if cm.is_gz_file(filepath):
        device = open(filepath, "rb")
    else:
        device = open(filepath, "r")

    std_reader = vcf.VCFReader(device, encoding="utf-8")

    if "VEP" in std_reader.metadata:
        if "CSQ" in std_reader.infos:
            device.close()
            return "vep"

    if "ANN" in std_reader.infos:
        device.close()
        return "snpeff"
    if "EFF" in std_reader.infos:
        device.close()
        return "snpeff3"


@contextmanager
def create_reader(filepath, vcf_annotation_parser=None):
    path = pathlib.Path(filepath)

    LOGGER.debug(
        "create_reader: PATH suffix %s, is_gz_file: %s",
        path.suffixes,
        cm.is_gz_file(filepath),
    )

    if ".vcf" in path.suffixes and ".gz" in path.suffixes:

        annotation_detected = vcf_annotation_parser or detect_vcf_annotation(filepath)

        reader = VcfReader(filepath, annotation_parser=annotation_detected)
        yield reader
        return

    if ".vcf" in path.suffixes:
        annotation_detected = detect_vcf_annotation(filepath)
        reader = VcfReader(filepath, annotation_parser=annotation_detected)
        yield reader
        return

    if {".tsv", ".csv", ".txt"} & set(path.suffixes):
        reader = CsvReader(filepath)
        yield reader
        return

    raise Exception("create_reader:: Could not choose parser for this file.")
