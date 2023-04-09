from typing import Tuple
import re

from gpanel256 import LOGGER




def quicksearch(query: str) -> dict:
    strategies = [
        parse_gene_query,
        parse_coords_query,

    ]
    for strat in strategies:
        parsed = strat(query)
        if parsed:
            return parsed
    return dict()


def parse_gene_query(query: str) -> dict:
    if not query:
        return dict()

    match = re.findall(r"^([\w-]+)$", query)
    if match:
        gene_name = match[0]

        gene_col_name = "gene"
        return {"$and": [{f"ann.{gene_col_name}": gene_name}]}
    else:
        return dict()


def parse_coords_query(query: str) -> bool:
    if not query:
        return ""

    match = re.findall(r"(\w+):(\d+)-(\d+)", query)

    if match:
        chrom, start, end = match[0]
        start = int(start)
        end = int(end)

        if end < start:
            return dict()
        return {"$and": [{"chr": chrom}, {"pos": {"$gte": start}}, {"pos": {"$lte": end}}]}
    else:
        return dict()

