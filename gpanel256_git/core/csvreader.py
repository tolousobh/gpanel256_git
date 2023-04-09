import csv

from .abstractreader import AbstractReader
from .annotationparser import VEP_ANNOTATION_DEFAULT_FIELDS, BaseParser


from gpanel256 import LOGGER


class CsvReader(AbstractReader):


    def __init__(self, device):
        super().__init__(device)

        first_line = device.readline()
        csv_dialect = csv.Sniffer().sniff(first_line)


        header = csv.Sniffer().has_header(first_line + device.readline())
        if not header:
            raise Exception("No header detected in the file; not a CSV file?")

        self.device.seek(0)
        self.csv_reader = csv.DictReader(self.device, dialect=csv_dialect)


        self.annotation_parser = BaseParser()
        self.annotation_parser.annotation_default_fields = VEP_ANNOTATION_DEFAULT_FIELDS


        self.ignored_columns = (
            "location",
            "allele",
            "#uploaded_variation",
            "given_ref",
            "used_ref",
        )

        self.fields = None

        LOGGER.debug("CsvReader::init: CSV fields found: %s", self.csv_reader.fieldnames)

    def __del__(self):
        del self.device

    def get_fields(self):

        LOGGER.debug("CsvReader::get_fields: called")
        if not self.fields:
            LOGGER.debug("CsvReader::get_fields: parse")
            self.fields = tuple(self.parse_fields())
        return self.fields

    def get_variants(self):

        yield from self.parse_variants()

    def get_samples(self):
        return []

    def parse_fields(self):
        yield {
            "name": "chr",
            "category": "variants",
            "description": "Chromosome",
            "type": "str",
            "constraint": "NOT NULL",
        }
        yield {
            "name": "pos",
            "category": "variants",
            "description": "Reference position, with the 1st base having position 1",
            "type": "int",
            "constraint": "NOT NULL",
        }
        yield {
            "name": "ref",
            "category": "variants",
            "description": "Reference base",
            "type": "str",
            "constraint": "NOT NULL",
        }
        yield {
            "name": "alt",
            "category": "variants",
            "description": "Alternative base",
            "type": "str",
            "constraint": "NOT NULL",
        }

        raw_fields = (
            field
            for field in self.csv_reader.fieldnames
            if field.lower() not in self.ignored_columns
        )

        self.annotation_parser.annotation_field_name = list()
        yield from self.annotation_parser.handle_descriptions(raw_fields)

    def parse_variants(self):


        def add_annotation_to_variant():

            if not annotation:
                return
            annotations = variant.get("annotations")
            if annotations:
                annotations.append(annotation)
            else:
                variant["annotations"] = [annotation]

        if self.annotation_parser.annotation_field_name is None:
            raise Exception("Cannot parse variant without parsing fields first")

        variants = dict()
        transcript_idx = 0
        for transcript_idx, row in enumerate(self.csv_reader, 1):

            chrom, pos = self.location_to_chr_pos(row["Location"])
            ref = row["GIVEN_REF"]
            alt = row["Allele"]

            if "USED_REF" in row:
                assert row["GIVEN_REF"] == row["USED_REF"], "GIVEN_REF != USED_REF"

            primary_key = (chrom, pos, ref, alt)
            variant = variants.get(primary_key, dict())

            annotation = dict()
            g = (key for key in row.keys() if key.lower() not in self.ignored_columns)
            for raw_key in g:
                lower_key = raw_key.lower()
                field_descript = VEP_ANNOTATION_DEFAULT_FIELDS.get(lower_key)
                if field_descript:
                    lower_key = field_descript["name"]

                annotation[lower_key] = row[raw_key]


            if variant:

                add_annotation_to_variant()
                continue

            variant["chr"], variant["pos"], variant["ref"], variant["alt"] = primary_key

            add_annotation_to_variant()



            variants[primary_key] = variant

        LOGGER.info(
            "CsvReader::parse_variants: transcripts %s, variants %s",
            transcript_idx,
            len(variants),
        )

        for variant in variants.values():
            yield dict(variant)

    def location_to_chr_pos(self, location: str):
        
        chrom, positions = location.split(":")
        pos = positions.split("-")[0]
        return chrom, pos

    def __repr__(self):
        return f"VEP Reader using {type(self.annotation_parser).__name__}"
