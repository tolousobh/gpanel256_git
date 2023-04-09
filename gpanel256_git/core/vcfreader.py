import vcf

from .abstractreader import AbstractReader, sanitize_field_name
from .annotationparser import VepParser, SnpEffParser
from gpanel256.commons import get_uncompressed_size

from gpanel256 import LOGGER


def _map(self, func, iterable, bad=[".", "", "NA", "-"]):

    def _convert(x):
        if x in bad:
            return None
        try:
            return func(x)
        except Exception as e:
            LOGGER.exception(e)
            return None

    return [_convert(x) for x in iterable]


vcf.Reader._map = _map



VCF_TYPE_MAPPING = {
    "Float": "float",
    "Integer": "int",
    "Flag": "bool",
    "String": "str",
    "Character": "str",
}


class VcfReader(AbstractReader):

    ANNOTATION_PARSERS = {
        "vep": VepParser,
        "snpeff": SnpEffParser,
        "snpeff3": SnpEffParser,
    }

    def __init__(self, filename, annotation_parser: str = None):

        super().__init__(filename)
        vcf_reader = vcf.VCFReader(filename=filename, strict_whitespace=True, encoding="utf-8")
        self.samples = vcf_reader.samples
        self.annotation_parser = None
        self.metadata = vcf_reader.metadata
        self._set_annotation_parser(annotation_parser)
        self.fields = None

        self.progress_every = 100
        self.total_bytes = vcf_reader.total_bytes()
        self.read_bytes = 0

    def progress(self) -> float:
        progress = self.read_bytes / self.total_bytes * 100
        return progress

    def get_fields(self):
        if self.fields is None:

            fields = tuple(self.parse_fields())
            for field in fields:
                field["name"] = sanitize_field_name(field["name"])

            if self.annotation_parser:

                self.fields = tuple(self.annotation_parser.parse_fields(fields))
            else:
                self.fields = fields
        return self.fields

    def get_variants(self):

        if self.fields is None:
            # This is a bad caching code ....
            self.get_fields()

        if self.annotation_parser:
            yield from self.annotation_parser.parse_variants(self.parse_variants())
        else:
            yield from self.parse_variants()

    def parse_variants(self):

        vcf_reader = vcf.VCFReader(
            filename=self.filename, strict_whitespace=True, encoding="utf-8"
        )

        format_fields = set(map(str.lower, vcf_reader.formats))
        format_fields.discard("gt")

        for i, record in enumerate(vcf_reader):

            self.read_bytes = vcf_reader.read_bytes()

            for index, alt in enumerate(record.ALT):
                variant = {
                    "chr": record.CHROM,
                    "pos": record.POS,
                    "ref": record.REF,
                    "alt": str(alt),
                    "rsid": record.ID,  # Avoid id column duplication in DB
                    "qual": record.QUAL,
                    "filter": "" if record.FILTER is None else ",".join(record.FILTER),
                }

                forbidden_field = ("chr", "pos", "ref", "alt", "rsid", "qual", "filter")

                for name in record.INFO:
                    if name.lower() not in forbidden_field:
                        if isinstance(record.INFO[name], list):
                            variant[name.lower()] = ",".join([str(i) for i in record.INFO[name]])
                        else:
                            variant[name.lower()] = record.INFO[name]

                if record.samples:
                    variant["samples"] = []
                    for sample in record.samples:
                        sample_data = {
                            "name": sample.sample,
                            "gt": -1 if sample.gt_type is None else sample.gt_type,
                        }

                        for gt_field in format_fields:
                            try:
                                value = sample[gt_field.upper()]
                                if isinstance(value, list):
                                    value = ",".join(str(i) for i in value)
                                sample_data[gt_field] = value
                            except AttributeError:

                                pass
                        variant["samples"].append(sample_data)

                yield variant

        self.read_bytes = self.total_bytes

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
        yield {
            "name": "rsid",
            "category": "variants",
            "description": "rsid unique identifier (see dbSNP)",
            "type": "str",
        }
        yield {
            "name": "qual",
            "category": "variants",
            "description": "Phred-scaled quality score for the assertion made in ALT:"
            "−10log10 prob(call in ALT is wrong).",
            "type": "int",
        }
        yield {
            "name": "filter",
            "category": "variants",
            "description": "Filter status: PASS if this position has passed all filters.",
            "type": "str",
        }

        vcf_reader = vcf.VCFReader(filename=self.filename, strict_whitespace=True, encoding="utf-8")

        for field_name, info in vcf_reader.infos.items():



            yield {
                "name": field_name.lower(),
                "category": "variants",
                "description": info.desc,
                "type": VCF_TYPE_MAPPING[info.type],
            }

        for field_name, info in vcf_reader.formats.items():
            description = info.desc
            field_type = VCF_TYPE_MAPPING[info.type]

            if field_name == "GT":
                description += " (0: homozygous_ref, 1: heterozygous, 2: homozygous_alt)"
                field_type = VCF_TYPE_MAPPING["Integer"]

            yield {
                "name": field_name.lower(),
                "category": "samples",
                "description": description,
                "type": field_type,
            }

    def get_samples(self):
        """Return list of samples (individual ids)."""
        return self.samples

    def _set_annotation_parser(self, parser: str):
        if parser in VcfReader.ANNOTATION_PARSERS:
            self.annotation_parser = VcfReader.ANNOTATION_PARSERS[parser]()
        else:
            self.annotation_parser = None

        if self.annotation_parser is None:
            LOGGER.info("Will not parse annotations")

    def __repr__(self):
        return f"VCF Reader using {type(self.annotation_parser).__name__}"

