
import sqlite3
from collections import defaultdict
import re
import logging
from sqlite3.dbapi2 import DatabaseError
import typing
from pkg_resources import parse_version
from functools import partial, lru_cache
import itertools as it
import numpy as np
import json
import os
import getpass

from typing import Dict, List, Callable, Iterable
from datetime import datetime

import gpanel256.constants as cst
import gpanel256.commons as cm

import gpanel256.core.querybuilder as qb
from gpanel256.core.sql_aggregator import StdevFunc
from gpanel256.core.reader import AbstractReader
from gpanel256.core.writer import AbstractWriter
from gpanel256.core.reader.pedreader import PedReader

from gpanel256 import LOGGER

import gpanel256.constants as cst

DEFAULT_SELECTION_NAME = cst.DEFAULT_SELECTION_NAME or "variants"
SAMPLES_SELECTION_NAME = cst.SAMPLES_SELECTION_NAME or "samples"
CURRENT_SAMPLE_SELECTION_NAME = cst.CURRENT_SAMPLE_SELECTION_NAME or "current_samples"
LOCKED_SELECTIONS = [DEFAULT_SELECTION_NAME, SAMPLES_SELECTION_NAME, CURRENT_SAMPLE_SELECTION_NAME]

PYTHON_TO_SQLITE = {
    "None": "NULL",
    "int": "INTEGER",
    "float": "REAL",
    "str": "TEXT",
    "bytes": "BLOB",
    "bool": "INTEGER",
}

SQLITE_TO_PYTHON = {
    "NULL": "None",
    "INTEGER": "int",
    "REAL": "float",
    "TEXT": "str",
    "BLOB": "bytes",
}

MANDATORY_FIELDS = [
    {
        "name": "chr",
        "type": "str",
        "category": "variants",
        "constraint": "DEFAULT 'unknown'",
        "description": "chromosom name",
    },
    {
        "name": "pos",
        "type": "int",
        "category": "variants",
        "constraint": "DEFAULT -1",
        "description": "variant position",
    },
    {
        "name": "ref",
        "type": "str",
        "category": "variants",
        "constraint": "DEFAULT 'N'",
        "description": "reference allele",
    },
    {
        "name": "alt",
        "type": "str",
        "category": "variants",
        "constraint": "DEFAULT 'N'",
        "description": "alternative allele",
    },
    {
        "name": "favorite",
        "type": "bool",
        "category": "variants",
        "constraint": "DEFAULT 0",
        "description": "favorite tag",
    },
    {
        "name": "comment",
        "type": "str",
        "category": "variants",
        "constraint": "DEFAULT ''",
        "description": "comment of variant",
    },
    {
        "name": "classification",
        "type": "int",
        "category": "variants",
        "constraint": "DEFAULT 0",
        "description": "ACMG score",
    },
    {
        "name": "tags",
        "type": "str",
        "category": "variants",
        "constraint": "DEFAULT ''",
        "description": "list of tags ",
    },
    {
        "name": "count_hom",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of homozygous genotypes (1/1)",
    },
    {
        "name": "count_het",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of heterozygous genotypes (0/1)",
    },
    {
        "name": "count_ref",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of homozygous genotypes (0/0)",
    },
    {
        "name": "count_none",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of none genotypes (./.)",
    },
    {
        "name": "count_tot",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of genotypes (all)",
    },
    {
        "name": "count_var",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of genotpyes heterozygous (0/1) or homozygous (1/1)",
    },
    {
        "name": "freq_var",
        "type": "float",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Frequency of variants for samples with genotypes (0/1 and 1/1)",
    },
    {
        "name": "count_validation_positive",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of validated genotypes",
    },
    {
        "name": "count_validation_negative",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of rejected genotypes",
    },
    {
        "name": "control_count_hom",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of homozygous genotypes (0/0) in control",
    },
    {
        "name": "control_count_het",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of homozygous genotypes (1/1) in control",
    },
    {
        "name": "control_count_ref",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of heterozygous genotypes (1/0) in control",
    },
    {
        "name": "case_count_hom",
        "type": "int",
        "category": "variants",
        "constraint": "DEFAULT 0",
        "description": "Number of homozygous genotypes (1/1) in case",
    },
    {
        "name": "case_count_het",
        "type": "int",
        "category": "variants",
        "constraint": "DEFAULT 0",
        "description": "Number of heterozygous genotypes (1/0) in case",
    },
    {
        "name": "case_count_ref",
        "type": "int",
        "category": "variants",
        "constraint": "DEFAULT 0",
        "description": "Number of homozygous genotypes (0/0) in case",
    },
    {
        "name": "is_indel",
        "type": "bool",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "True if variant is an indel",
    },
    {
        "name": "is_snp",
        "type": "bool",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "True if variant is a snp",
    },
    {
        "name": "annotation_count",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "variants",
        "description": "Number of transcript",
    },
    ## SAMPLES
    {
        "name": "classification",
        "type": "int",
        "constraint": "DEFAULT 0",
        "category": "samples",
        "description": "classification",
    },
    {
        "name": "comment",
        "type": "str",
        "constraint": "DEFAULT ''",
        "category": "samples",
        "description": "comment of variant",
    },
    {
        "name": "tags",
        "type": "str",
        "category": "samples",
        "constraint": "DEFAULT ''",
        "description": "list of tags ",
    },
    {
        "name": "gt",
        "type": "int",
        "constraint": "DEFAULT -1",
        "category": "samples",
        "description": "Genotype",
    },
]





def get_sql_connection(filepath: str) -> sqlite3.Connection:

    connection = sqlite3.connect(filepath)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row
    foreign_keys_status = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    LOGGER.debug("get_sql_connection:: foreign_keys state: %s", foreign_keys_status)
    assert foreign_keys_status == 1, "Foreign keys can't be activated :("

    def regexp(expr, item):
        return re.search(expr, str(item)) is not None

    connection.create_function("REGEXP", 2, regexp)
    connection.create_function("current_user", 0, lambda: getpass.getuser())
    connection.create_aggregate("STD", 1, StdevFunc)

    if LOGGER.getEffectiveLevel() == logging.DEBUG:
        sqlite3.enable_callback_tracebacks(True)


    return connection


def get_database_file_name(conn: sqlite3.Connection) -> str:
    return conn.execute("PRAGMA database_list").fetchone()["file"]


def schema_exists(conn: sqlite3.Connection) -> bool:
    query = "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
    return conn.execute(query).fetchone()[0] > 0


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    c = conn.cursor()
    c.execute(f"SELECT name FROM sqlite_master WHERE name = '{name}'")
    return c.fetchone() != None


def drop_table(conn, table_name: str):
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()


def clear_table(conn: sqlite3.Connection, table_name: str):
    cursor = conn.cursor()
    cursor.execute(f"DELETE  FROM {table_name}")
    conn.commit()


def get_table_columns(conn: sqlite3.Connection, table_name: str):
    return [c[1] for c in conn.execute(f"pragma table_info({table_name})") if c[1] != "id"]


def alter_table(conn: sqlite3.Connection, table_name: str, fields: list):
    for field in fields:

        name = field["name"]
        p_type = field["type"]
        s_type = PYTHON_TO_SQLITE.get(p_type, "TEXT")
        constraint = field.get("constraint", "")
        sql = f"ALTER TABLE {table_name} ADD COLUMN `{name}` {s_type} {constraint}"

        try:
            conn.execute(sql)

        except sqlite3.OperationalError as e:
            LOGGER.error(e)

    conn.commit()


def alter_table_from_fields(conn: sqlite3.Connection, fields: list):

    tables = ["variants", "annotations", "genotypes"]

    if not schema_exists(conn):
        LOGGER.error("CANNOT ALTER TABLE. NO SCHEMA AVAILABLE")
        return

    for table in tables:

        category = "samples" if table == "genotypes" else table

        # Get local columns names
        local_col_names = set(get_table_columns(conn, table))

        # get new fields which are not in local
        new_fields = [
            i for i in fields if i["category"] == category and i["name"] not in local_col_names
        ]

        if new_fields:
            alter_table(conn, table, new_fields)


def count_query(conn: sqlite3.Connection, query: str) -> int:
    return conn.execute(f"SELECT COUNT(*) as count FROM ({query})").fetchone()[0]



def get_stats_info(conn, field, source="variants", filters={}):
    pass


def get_field_info(conn, field, source="variants", filters={}, metrics=["mean", "std"]):

    metric_functions = {
        "count": len,
        "mean": np.mean,
        "std": np.std,
        "min": lambda ar: np.quantile(ar, 0.0),
        "q1": lambda ar: np.quantile(ar, 0.25),
        "median": lambda ar: np.quantile(ar, 0.5),
        "q3": lambda ar: np.quantile(ar, 0.75),
        "max": lambda ar: np.quantile(ar, 1.0),
    }

    conn.row_factory = None
    query = qb.build_sql_query(conn, [field], source, filters, limit=None)

    data = [i[0] for i in conn.execute(query)]

    results = {}
    for metric in metrics:
        if metric in metric_functions:
            value = metric_functions[metric](data)
            results[metric] = value

        if isinstance(metric, tuple) and len(metric) == 2:
            metric_name, metric_func = metric
            if callable(metric_func):
                value = metric_func(data)
                results[metric_name] = value

    conn.row_factory = sqlite3.Row

    return results


def get_indexed_fields(conn: sqlite3.Connection) -> List[tuple]:
    indexed_fields = [
        dict(res)["name"]
        for res in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    ]
    result = []
    find_indexed = re.compile(r"idx_(variants|annotations|samples)_(.+)")
    for index in indexed_fields:
        matches = find_indexed.findall(index)
        if matches and len(matches[0]) == 2:
            category, field_name = matches[0]
            result.append((category, field_name))
    return result


def remove_indexed_field(conn: sqlite3.Connection, category: str, field_name: str):
    conn.execute(f"DROP INDEX IF EXISTS idx_{category}_{field_name}")
    conn.commit()


def create_indexes(
    conn: sqlite3.Connection,
    indexed_variant_fields: list = None,
    indexed_annotation_fields: list = None,
    indexed_sample_fields: list = None,
    progress_callback: Callable = None,
):

    if progress_callback:
        progress_callback("Create selection index ")

    create_selections_indexes(conn)

    if progress_callback:
        progress_callback("Create variants index ")

    create_variants_indexes(conn, indexed_variant_fields)

    if progress_callback:
        progress_callback("Create samples index ")

    create_samples_indexes(conn, indexed_sample_fields)

    try:
        # Some databases have not annotations table
        if progress_callback:
            progress_callback("Create annotation index  ")
        create_annotations_indexes(conn, indexed_annotation_fields)

    except sqlite3.OperationalError as e:
        LOGGER.debug("create_indexes:: sqlite3.%s: %s", e.__class__.__name__, str(e))


def get_clean_fields(fields: Iterable[dict] = None) -> Iterable[dict]:

    if fields is None:
        fields = []

    required_fields = {(f["category"], f["name"]): f for f in MANDATORY_FIELDS}
    input_fields = {(f["category"], f["name"]): f for f in fields}

    required_fields.update(input_fields)

    for field in required_fields.values():
        yield field


def get_accepted_fields(fields: Iterable[dict], ignored_fields: Iterable[dict]) -> Iterable[dict]:

    ignored_keys = {(f["category"], f["name"]) for f in ignored_fields}
    return list(filter(lambda x: (x["category"], x["name"]) not in ignored_keys, fields))


def get_clean_variants(variants: Iterable[dict]) -> Iterable[dict]:

    for variant in variants:
        variant["is_indel"] = len(variant["ref"]) != len(variant["alt"])
        variant["is_snp"] = len(variant["ref"]) == len(variant["alt"])
        variant["annotation_count"] = len(variant["annotations"]) if "annotations" in variant else 0

        yield variant


def create_table_project(conn: sqlite3.Connection):

    conn.execute("CREATE TABLE projects (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()


def update_project(conn: sqlite3.Connection, project: dict):
    conn.executemany(
        "INSERT OR REPLACE INTO projects (key, value) VALUES (?, ?)",
        list(project.items()),
    )
    conn.commit()


def get_project(conn: sqlite3.Connection) -> dict:
    g = (dict(data) for data in conn.execute("SELECT key, value FROM projects"))
    return {data["key"]: data["value"] for data in g}




def create_table_metadatas(conn: sqlite3.Connection):

    conn.execute("CREATE TABLE metadatas (key TEXT PRIMARY KEY, value TEXT)")


def update_metadatas(conn: sqlite3.Connection, metadatas: dict):

    if metadatas:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO metadatas (key,value) VALUES (?,?)",
            list(metadatas.items()),
        )

        conn.commit()


def get_metadatas(conn: sqlite3.Connection) -> dict:
    conn.row_factory = sqlite3.Row
    g = (dict(data) for data in conn.execute("SELECT key, value FROM metadatas"))
    return {data["key"]: data["value"] for data in g}


def create_table_selections(conn: sqlite3.Connection):

    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE selections (
        id INTEGER PRIMARY KEY ASC,
        name TEXT UNIQUE, count INTEGER, query TEXT, description TEXT
        )"""
    )
    cursor.execute(
        """CREATE TABLE selection_has_variant (
        variant_id INTEGER NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
        selection_id INTEGER NOT NULL REFERENCES selections(id) ON DELETE CASCADE,
        PRIMARY KEY (variant_id, selection_id)
        )"""
    )
    conn.commit()


def create_selections_indexes(conn: sqlite3.Connection):
    conn.execute("CREATE UNIQUE INDEX idx_selections ON selections (name)")


def create_selection_has_variant_indexes(conn: sqlite3.Connection):
    conn.execute(
        "CREATE INDEX `idx_selection_has_variant` ON selection_has_variant (`selection_id`)"
    )


def insert_selection(
    conn: sqlite3.Connection,
    query: str,
    name: str = "no_name",
    count: int = 0,
    description: str = None,
) -> int:

    if name == DEFAULT_SELECTION_NAME and description is None:
        description = "All variants"

    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR REPLACE INTO selections (name, count, query, description) VALUES (?,?,?,?)",
        (name, count, query, description),
    )

    conn.commit()

    return cursor.lastrowid



def insert_selection_from_samples(
    conn: sqlite3.Connection,
    samples: list,
    name: str = SAMPLES_SELECTION_NAME,
    gt_min: int = 0,
    force: bool = True,
    description: str = None,
) -> int:

    samples_clause = "(" + ",".join([f"'{i}'" for i in samples]) + ")"
    ids = ",".join(
        [
            str(dict(rec)["id"])
            for rec in conn.execute(f"SELECT id FROM samples WHERE name in {samples_clause}")
        ]
    )

    query = f"""SELECT distinct(id) FROM variants INNER JOIN genotypes ON genotypes.variant_id = variants.id 
    WHERE genotypes.sample_id IN ({ids}) AND genotypes.gt > {gt_min}"""

    if description == None and (
        name == SAMPLES_SELECTION_NAME or name == CURRENT_SAMPLE_SELECTION_NAME
    ):
        description = ",".join(samples)

    selections = get_selections(conn)
    query_in_db = None
    for s in selections:
        if s["name"] == name:
            query_in_db = s["query"]

    if query_in_db != query or force:
        return insert_selection_from_sql(conn=conn, query=query, name=name, description=description)
    else:
        return None


def insert_selection_from_sql(
    conn: sqlite3.Connection,
    query: str,
    name: str,
    count: int = None,
    from_selection: bool = False,
    description: str = None,
) -> int:
    cursor = conn.cursor()

    if count is None:
        count = count_query(conn=conn, query=query)

    selection_id = insert_selection(
        conn=conn, query=query, name=name, count=count, description=description
    )

    try:
        cursor.execute("""DROP INDEX idx_selection_has_variant""")
    except sqlite3.OperationalError:
        pass


    if from_selection:
        q = f"""
        INSERT INTO selection_has_variant
        SELECT DISTINCT variant_id, {selection_id} FROM ({query})
        """
    else:
        q = f"""
        INSERT INTO selection_has_variant
        SELECT DISTINCT id, {selection_id} FROM ({query})
        """

    cursor.execute(q)
    affected_rows = cursor.rowcount


    create_selection_has_variant_indexes(cursor)

    if affected_rows:
        conn.commit()
        return selection_id
    conn.rollback()
    delete_selection(conn, selection_id)
    return None




def get_selections(conn: sqlite3.Connection) -> List[dict]:
    conn.row_factory = sqlite3.Row
    return (dict(data) for data in conn.execute("SELECT * FROM selections"))




def delete_selection_by_name(conn: sqlite3.Connection, name: str):

    if name == "variants":
        LOGGER.error("Cannot remove the default selection 'variants'")
        return

    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM selections WHERE name = ?", (name,))
    conn.commit()
    return cursor.rowcount






def create_table_wordsets(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE wordsets (
        id INTEGER PRIMARY KEY ASC,
        name TEXT,
        value TEXT,
        UNIQUE (name, value)
        )"""
    )

    conn.commit()




def intersect_variants(query1, query2, **kwargs):
    return f"""SELECT * FROM ({query1} INTERSECT {query2})"""


def union_variants(query1, query2, **kwargs):
    return f"""{query1} UNION {query2}"""


def subtract_variants(query1, query2, **kwargs):
    return f"""{query1} EXCEPT {query2}"""



def create_table_fields(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE fields
        (id INTEGER PRIMARY KEY, name TEXT, category TEXT, type TEXT, description TEXT, UNIQUE(name, category))
        """
    )
    conn.commit()


def insert_field(conn, name="no_name", category="variants", field_type="text", description=""):

    insert_fields(
        conn,
        [
            {
                "name": name,
                "category": category,
                "type": field_type,
                "description": description,
            }
        ],
    )


def insert_fields(conn: sqlite3.Connection, data: list):
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT OR IGNORE INTO fields (name,category,type,description)
        VALUES (:name,:category,:type,:description)
        """,
        data,
    )
    conn.commit()



def get_fields(conn):
    conn.row_factory = sqlite3.Row
    return tuple(dict(data) for data in conn.execute("SELECT * FROM fields"))




def get_field_by_name(conn, field_name: str):
    conn.row_factory = sqlite3.Row
    field_data = conn.execute("SELECT * FROM fields WHERE name = ? ", (field_name,)).fetchone()
    return dict(field_data) if field_data else None


def get_field_range(conn, field_name: str, sample_name=None):
    field = get_field_by_name(conn, field_name)
    if not field:
        return None

    table = field["category"]  # variants, or annotations or samples
    if table == "samples":
        if not sample_name:
            raise ValueError("Pass sample parameter for sample fields")
        query = f"""SELECT min({field_name}), max({field_name})
        FROM genotypes
        JOIN samples ON genotypes.sample_id = samples.id
        WHERE samples.name='{sample_name}'
        """
    else:
        query = f"SELECT min({field_name}), max({field_name}) FROM {table}"

    result = tuple(conn.execute(query).fetchone())
    if result in ((None, None), ("", "")):
        return None

    return result


def get_field_unique_values(conn, field_name: str, like: str = None, limit=None):

    if field_name.startswith("ann."):
        field_name = field_name.replace("ann.", "")

    if field_name.startswith("samples."):
        _, *_, field = field_name.split(".")
        field_name = field

    field = get_field_by_name(conn, field_name)
    if not field:
        return []
    table = field["category"]  # variants, or annotations or samples

    if table == "samples":
        query = f""" SELECT DISTINCT `{field_name}` FROM genotypes """

    elif table == "annotations":
        query = f""" SELECT DISTINCT `{field_name}` FROM annotations """

    else:
        query = f"SELECT DISTINCT `{field_name}` FROM {table}"

    if like:
        query += f" WHERE `{field_name}` LIKE '{like}'"

    if limit:
        query += " LIMIT " + str(limit)

    return [i[field_name] for i in conn.execute(query)]




def create_table_annotations(conn: sqlite3.Connection, fields: List[dict]):

    schema = ",".join([f'`{field["name"]}` {field["type"]}' for field in fields])

    if not schema:
        schema = "gene TEXT, transcript TEXT"
        LOGGER.debug("create_table_annotations:: No annotation fields detected! => Fallback")


    cursor = conn.cursor()

    cursor.execute(
        f"""CREATE TABLE annotations (variant_id 
        INTEGER REFERENCES variants(id) ON UPDATE CASCADE,
         {schema})

        """
    )

    conn.commit()


def create_annotations_indexes(conn, indexed_annotation_fields=None):

    conn.execute("CREATE INDEX IF NOT EXISTS `idx_annotations` ON annotations (`variant_id`)")

    if indexed_annotation_fields is None:
        return
    for field in indexed_annotation_fields:

        LOGGER.debug(
            f"CREATE INDEX IF NOT EXISTS `idx_annotations_{field}` ON annotations (`{field}`)"
        )

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS `idx_annotations_{field}` ON annotations (`{field}`)"
        )


def get_annotations(conn, variant_id: int):
    """Get variant annotation for the variant with the given id"""
    conn.row_factory = sqlite3.Row
    for annotation in conn.execute(f"SELECT * FROM annotations WHERE variant_id = {variant_id}"):
        yield dict(annotation)




def create_table_variants(conn: sqlite3.Connection, fields: List[dict]):
    cursor = conn.cursor()

    schema = ",".join(
        [
            f'`{field["name"]}` {PYTHON_TO_SQLITE.get(field["type"],"TEXT")} {field.get("constraint", "")}'
            for field in fields
            if field["name"]
        ]
    )


    LOGGER.debug("create_table_variants:: schema: %s", schema)

    cursor.execute(
        f"""CREATE TABLE variants (id INTEGER PRIMARY KEY, {schema},
        UNIQUE (chr,pos,ref,alt))"""
    )

    conn.commit()


def create_variants_indexes(conn, indexed_fields={"pos", "ref", "alt"}):

    conn.execute(
        "CREATE INDEX IF NOT EXISTS `idx_genotypes` ON genotypes (`variant_id`, `sample_id`)"
    )

    conn.execute("CREATE INDEX IF NOT EXISTS `idx_genotypes_sample_id` ON genotypes (`sample_id`)")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS `idx_genotypes_variant_id` ON genotypes (`variant_id`)"
    )

    conn.execute("CREATE INDEX IF NOT EXISTS `idx_genotypes_gt` ON genotypes (`gt`)")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS `idx_genotypes_classification` ON genotypes (`classification`)"
    )

    for field in indexed_fields:
        conn.execute(f"CREATE INDEX IF NOT EXISTS `idx_variants_{field}` ON variants (`{field}`)")


def get_variant(
    conn: sqlite3.Connection, variant_id: int, with_annotations=False, with_samples=False
):
    conn.row_factory = sqlite3.Row
    variant = dict(
        conn.execute(f"SELECT * FROM variants WHERE variants.id = {variant_id}").fetchone()
    )

    variant["annotations"] = []
    if with_annotations:
        variant["annotations"] = [
            dict(annotation)
            for annotation in conn.execute(
                f"SELECT * FROM annotations WHERE variant_id = {variant_id}"
            )
        ]

    variant["samples"] = []
    if with_samples:
        variant["samples"] = [
            dict(sample)
            for sample in conn.execute(
                f"""SELECT samples.name, genotypes.* FROM samples
                LEFT JOIN genotypes on samples.id = genotypes.sample_id
                WHERE variant_id = {variant_id} """
            )
        ]

    return variant


def update_variant(conn: sqlite3.Connection, variant: dict):
    if "id" not in variant:
        raise KeyError("'id' key is not in the given variant <%s>" % variant)

    unzip = lambda l: list(zip(*l))

    placeholders, values = unzip(
        [(f"`{key}` = ? ", value) for key, value in variant.items() if key != "id"]
    )
    query = "UPDATE variants SET " + ",".join(placeholders) + f" WHERE id = {variant['id']}"
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()


def get_variants_count(conn: sqlite3.Connection):
    return count_query(conn, "variants")


def get_variant_occurences(conn: sqlite3.Connection, variant_id: int):

    for rec in conn.execute(
        f"""
        SELECT samples.name, genotypes.* FROM genotypes 
        INNER JOIN samples ON samples.id = genotypes.sample_id 
        WHERE genotypes.variant_id = {variant_id} AND genotypes.gt > 0"""
    ):
        yield dict(rec)



def get_sample_variant_classification_count(
    conn: sqlite3.Connection, sample_id: int, classification: int
):
    r = conn.execute(
        f"SELECT COUNT(*) FROM genotypes sv WHERE sv.sample_id={sample_id} AND classification = {classification}"
    ).fetchone()[0]
    return int(r)


def get_sample_variant_classification(
    conn: sqlite3.Connection, sample_id: int = None, variant_id: int = None
):
    where_clause = " 1=1 "
    if sample_id:
        where_clause += f" AND genotypes.sample_id={sample_id} "
    if variant_id:
        where_clause += f" AND genotypes.variant_id={variant_id} "
    r = conn.execute(
        f"""
        SELECT samples.name, genotypes.* 
        FROM genotypes
        INNER JOIN samples ON samples.id = genotypes.sample_id 
        WHERE {where_clause}
        """
    )
    return (dict(data) for data in r)


def get_samples_from_query(conn: sqlite3.Connection, query: str):

    if not query:
        return get_samples(conn)

    or_list = []
    for word in query.split():
        if ":" not in word:
            word = f"name:{word}"
        for i in re.findall(r"(.+):(.+)", word):
            if "," in i[1]:
                key, val = i[0], f"{i[1]}"
                val = ",".join([f"'{i}'" for i in val.split(",")])
                or_list.append(f"{key} IN ({val})")
            else:
                key, val = i[0], i[1]
                if key in ("name", "tags"):
                    or_list.append(f"{key} LIKE '%{val}%'")
                else:
                    or_list.append(f"{key} = '{val}'")

    sql_query = f"SELECT * FROM samples WHERE {' OR '.join(or_list)}"
    # Suppose conn.row_factory = sqlite3.Row

    return (dict(data) for data in conn.execute(sql_query))


def get_variants(
    conn: sqlite3.Connection,
    fields,
    source="variants",
    filters={},
    order_by=None,
    order_desc=True,
    limit=50,
    offset=0,
    group_by={},
    having={},  # {"op":">", "value": 3  }
    **kwargs,
):


    query = qb.build_sql_query(
        conn,
        fields=fields,
        source=source,
        filters=filters,
        order_by=order_by,
        order_desc=order_desc,
        limit=limit,
        offset=offset,
        group_by=group_by,
        having=having,
        **kwargs,
    )

    for i in conn.execute(query):

        yield {k.replace("(", "[").replace(")", "]"): v for k, v in dict(i).items()}


def get_variants_tree(
    conn: sqlite3.Connection,
    **kwargs,
):
    pass



def update_variants_counts(
    conn: sqlite3.Connection,
    progress_callback: Callable = None,
):

    if progress_callback:
        progress_callback("Variants count_het")
    conn.execute(
        """
        UPDATE variants
        SET count_het = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes
            WHERE genotypes.gt = 1
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    if progress_callback:
        progress_callback("Variants count_hom")
    conn.execute(
        """
        UPDATE variants
        SET count_hom = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes
            WHERE genotypes.gt = 2
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    if progress_callback:
        progress_callback("Variants count_ref")
    conn.execute(
        """
        UPDATE variants
        SET count_ref = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes
            WHERE genotypes.gt = 0
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    if progress_callback:
        progress_callback("Variants count_var")
    conn.execute(
        """
        UPDATE variants
        SET count_var = count_het + count_hom
        """
    )

    sample_count = conn.execute("SELECT COUNT(id) FROM samples").fetchone()[0]

    if progress_callback:
        progress_callback("Variants count_tot")
    conn.execute(
        f"""
        UPDATE variants
        SET count_tot = {sample_count}
        """
    )

    if progress_callback:
        progress_callback("Variants count_none")
    conn.execute(
        f"""
        UPDATE variants
        SET count_none = count_tot - count_var
        """
    )

    if progress_callback:
        progress_callback("Variants freq_var")
    conn.execute(
        """
        UPDATE variants
        SET freq_var = ( cast ( ( (count_hom * 2) + count_het ) as real) / ( cast ( (count_tot * 2) as real ) ) )
        """
    )

    conn.commit()


    pheno_count = conn.execute(
        "SELECT COUNT(phenotype) FROM samples WHERE phenotype > 0"
    ).fetchone()[0]
    if pheno_count == 0:
        LOGGER.warning("No phenotype. Do not compute case/control count")
        return

    conn.execute(
        """
        UPDATE variants
        SET case_count_hom = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes, samples
            WHERE genotypes.sample_id=samples.id AND genotypes.gt = 2 AND samples.phenotype=2
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    conn.execute(
        """
        UPDATE variants
        SET case_count_het = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes, samples
            WHERE genotypes.sample_id=samples.id AND genotypes.gt = 1 AND samples.phenotype=2
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    conn.execute(
        """
        UPDATE variants
        SET case_count_ref = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes, samples
            WHERE genotypes.sample_id=samples.id AND genotypes.gt = 0 AND samples.phenotype=2
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    conn.execute(
        """
        UPDATE variants
        SET control_count_hom = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes, samples
            WHERE genotypes.sample_id=samples.id AND genotypes.gt = 2 AND samples.phenotype=1
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    conn.execute(
        """
        UPDATE variants
        SET control_count_het = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes, samples
            WHERE genotypes.sample_id=samples.id AND genotypes.gt = 1 AND samples.phenotype=1
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    conn.execute(
        """
        UPDATE variants
        SET control_count_ref = geno.count
        FROM (SELECT variant_id, count(sample_id) as count
            FROM genotypes, samples
            WHERE genotypes.sample_id=samples.id AND genotypes.gt = 0 AND samples.phenotype=1
            GROUP BY variant_id) as geno
        WHERE id = geno.variant_id;
        """
    )

    conn.commit()


def insert_variants(
    conn: sqlite3.Connection,
    variants: List[dict],
    total_variant_count: int = None,
    progress_every: int = 1000,
    progress_callback: Callable = None,
):

    variants_local_fields = set(get_table_columns(conn, "variants"))
    annotations_local_fields = set(get_table_columns(conn, "annotations"))
    samples_local_fields = set(get_table_columns(conn, "genotypes"))

    samples_map = {sample["name"]: sample["id"] for sample in get_samples(conn)}

    progress = -1
    errors = 0
    cursor = conn.cursor()
    batches = []
    total = 0

    RETURNING_ENABLE = parse_version(sqlite3.sqlite_version) >= parse_version("3.35.0 ")

    for variant_count, variant in enumerate(variants):

        variant_fields = {i for i in variant.keys() if i not in ("samples", "annotations")}

        common_fields = variant_fields & variants_local_fields

        query_fields = ",".join((f"`{i}`" for i in common_fields))
        query_values = ",".join((f"?" for i in common_fields))
        query_datas = [variant[i] for i in common_fields]


        if RETURNING_ENABLE:
            query = f"""INSERT INTO variants ({query_fields}) VALUES ({query_values}) ON CONFLICT (chr,pos,ref,alt) 
            DO UPDATE SET ({query_fields}) = ({query_values}) RETURNING id
            """
            res = cursor.execute(query, query_datas * 2).fetchone()
            variant_id = dict(res)["id"]
        else:
            query = f"""INSERT INTO variants ({query_fields}) VALUES ({query_values}) ON CONFLICT (chr,pos,ref,alt) 
            DO UPDATE SET ({query_fields}) = ({query_values})
            """
            cursor.execute(query, query_datas * 2)

            chrom = variant["chr"]
            pos = variant["pos"]
            ref = variant["ref"]
            alt = variant["alt"]

            variant_id = conn.execute(
                f"SELECT id FROM variants where chr='{chrom}' AND pos = {pos} AND ref='{ref}' AND alt='{alt}'"
            ).fetchone()[0]

        total += 1


        if variant_id == 0:
            LOGGER.debug(
            )
            errors += 1
            total -= 1
            continue

        if "annotations" in variant:
            cursor.execute(f"DELETE FROM annotations WHERE variant_id ={variant_id}")
            for ann in variant["annotations"]:

                ann["variant_id"] = variant_id
                common_fields = annotations_local_fields & ann.keys()
                query_fields = ",".join((f"`{i}`" for i in common_fields))
                query_values = ",".join((f"?" for i in common_fields))
                query_datas = [ann[i] for i in common_fields]
                query = (
                    f"INSERT OR REPLACE INTO annotations ({query_fields}) VALUES ({query_values})"
                )

                cursor.execute(query, query_datas)


        if "samples" in variant:
            for sample in variant["samples"]:
                if sample["name"] in samples_map:

                    sample["variant_id"] = int(variant_id)
                    sample["sample_id"] = int(samples_map[sample["name"]])

                    sample["gt"] = sample.get("gt", -1)
                    if sample["gt"] < 0:  # Allow genotype 1,2,3,4,5,... ( for other species )
                        # remove gt if exists
                        query_remove = f"""DELETE FROM genotypes WHERE variant_id={sample["variant_id"]} AND sample_id={sample["sample_id"]}"""
                        cursor.execute(query_remove)
                    else:
                        common_fields = samples_local_fields & sample.keys()
                        query_fields = ",".join((f"`{i}`" for i in common_fields))
                        query_values = ",".join((f"?" for i in common_fields))
                        query_datas = [sample[i] for i in common_fields]
                        query = f"""INSERT INTO genotypes ({query_fields}) VALUES ({query_values}) ON CONFLICT (variant_id, sample_id)
                        DO UPDATE SET ({query_fields}) = ({query_values})
                        """
                        cursor.execute(query, query_datas * 2)

        if progress_callback and variant_count != 0 and variant_count % progress_every == 0:
            progress_callback(f"{variant_count} variants inserted.")

    conn.commit()

    if progress_callback:
        progress_callback(f"{total} variant(s) has been inserted with {errors} error(s)")

es.

    true_total = conn.execute("SELECT COUNT(*) FROM variants").fetchone()[0]
    insert_selection(conn, query="", name=DEFAULT_SELECTION_NAME, count=true_total)










## samples table ===============================================================


def create_table_samples(conn, fields=[]):

    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE samples (
        id INTEGER PRIMARY KEY ASC,
        name TEXT,
        family_id TEXT DEFAULT 'fam',
        father_id INTEGER DEFAULT 0,
        mother_id INTEGER DEFAULT 0,
        sex INTEGER DEFAULT 0,
        phenotype INTEGER DEFAULT 0,
        classification INTEGER DEFAULT 0,
        tags TEXT DEFAULT '',
        comment TEXT DEFAULT '',
        count_validation_positive_variant INTEGER DEFAULT 0,
        count_validation_negative_variant INTEGER DEFAULT 0,
        UNIQUE (name, family_id)
        )"""
    )
    conn.commit()

    fields = list(fields)

    schema = ",".join(
        [
            f'`{field["name"]}` {field["type"]} {field.get("constraint", "")}'
            for field in fields
            if field["name"]
        ]
    )

    cursor.execute(
        f"""CREATE TABLE genotypes  (
        sample_id INTEGER NOT NULL,
        variant_id INTEGER NOT NULL,
        {schema},
        PRIMARY KEY (sample_id, variant_id),
        FOREIGN KEY (sample_id) REFERENCES samples (id)
          ON DELETE CASCADE
          ON UPDATE NO ACTION
        ) 
       """
    )


    conn.commit()


def create_samples_indexes(conn, indexed_samples_fields=None):
    """Create indexes on the "samples" table"""
    if indexed_samples_fields is None:
        return

    for field in indexed_samples_fields:
        conn.execute(f"CREATE INDEX IF NOT EXISTS `idx_samples_{field}` ON genotypes (`{field}`)")


def insert_sample(conn, name="no_name"):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO samples (name) VALUES (?)", [name])
    conn.commit()
    return cursor.lastrowid


def insert_samples(conn, samples: list, import_id: str = None, import_vcf: str = None):

    current_date = datetime.today().strftime("%Y%m%d-%H%M%S")

    import_vcf_tag = None
    if import_vcf:
        import_vcf_tag = "importVCF#" + import_vcf

    import_date_tag = None
    import_date_tag = "importDATE#" + current_date

    import_id_tag = None
    if not import_id:
        import_id = current_date
    import_id_tag = "importID#" + import_id

    cursor = conn.cursor()

    for sample in samples:
        cursor.execute(f"INSERT OR IGNORE INTO samples (name) VALUES ('{sample}') ")
        if import_vcf_tag:
            cursor.execute(
                f"UPDATE samples SET tags = '{import_vcf_tag}' WHERE name = '{sample}' AND tags = '' "
            )
            cursor.execute(
                f"UPDATE samples SET tags = tags || ',' || '{import_vcf_tag}' WHERE name = '{sample}' AND tags != '' AND ',' || tags || ',' NOT LIKE '%,{import_vcf_tag},%' "
            )
        if import_date_tag:
            cursor.execute(
                f"UPDATE samples SET tags = '{import_date_tag}' WHERE name = '{sample}' AND tags = '' "
            )
            cursor.execute(
                f"UPDATE samples SET tags = tags || ',' || '{import_date_tag}' WHERE name = '{sample}' AND tags != '' AND ',' || tags || ',' NOT LIKE '%,{import_date_tag},%' "
            )
        if import_id_tag:
            cursor.execute(
                f"UPDATE samples SET tags = '{import_id_tag}' WHERE name = '{sample}' AND tags = '' "
            )
            cursor.execute(
                f"UPDATE samples SET tags = tags || ',' || '{import_id_tag}' WHERE name = '{sample}' AND tags != '' AND ',' || tags || ',' NOT LIKE '%,{import_id_tag},%' "
            )

    conn.commit()


def get_samples(conn: sqlite3.Connection):
    conn.row_factory = sqlite3.Row
    return (dict(data) for data in conn.execute("SELECT * FROM samples"))


def search_samples(conn: sqlite3.Connection, name: str, families=[], tags=[], classifications=[]):

    query = """
    SELECT * FROM samples
    """

    clauses = []

    if name:
        clauses.append(f"name LIKE '%{name}%'")

    if families:
        families_clause = ",".join(f"'{i}'" for i in families)
        clauses.append(f"family_id IN ({families_clause})")

    if classifications:
        classification_clause = ",".join(f"{i}" for i in classifications)
        clauses.append(f" classification IN ({classification_clause})")


    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    print(query)
    for sample in conn.execute(query):
        yield dict(sample)


def get_sample(conn: sqlite3.Connection, sample_id: int):
    """Get samples information from a specific id

    Args:
        conn (sqlite3.Connection): sqlite3.Connextion
        sample_id (int): sample table id
    """

    return dict(conn.execute(f"SELECT * FROM samples WHERE id = {sample_id}").fetchone())


def get_sample_annotations(conn, variant_id: int, sample_id: int):
    """Get samples for given sample id and variant id"""
    conn.row_factory = sqlite3.Row
    return dict(
        conn.execute(
            f"SELECT * FROM genotypes WHERE variant_id = {variant_id} and sample_id = {sample_id}"
        ).fetchone()
    )


def get_sample_nb_genotype_by_classification(conn, sample_id: int):
    """Get number of genotype by classification for given sample id"""
    conn.row_factory = sqlite3.Row
    return dict(
        conn.execute(
            f"SELECT genotypes.classification as classification, count(genotypes.variant_id) as nb_genotype FROM genotypes WHERE genotypes.sample_id = '{sample_id}' GROUP BY genotypes.classification"
        )
    )


def get_if_sample_has_classified_genotypes(conn, sample_id: int):
    """Get if sample id has classificed genotype (>0)"""
    conn.row_factory = sqlite3.Row
    res = conn.execute(
        f"SELECT 1 as variant FROM genotypes WHERE genotypes.sample_id = '{sample_id}' AND classification > 0 LIMIT 1"
    ).fetchone()
    if res:
        return True
    else:
        return False


def get_genotypes(conn, variant_id: int, fields: List[str] = None, samples: List[str] = None):
    fields = fields or ["gt"]

    sql_fields = ",".join([f"sv.{f}" for f in fields])

    query = f"""SELECT sv.sample_id, sv.variant_id, samples.name , {sql_fields} FROM samples
    LEFT JOIN genotypes sv 
    ON sv.sample_id = samples.id AND sv.variant_id = {variant_id}  """

    conditions = []

    if samples:
        sample_clause = ",".join([f"'{s}'" for s in samples])
        query += f"WHERE samples.name IN ({sample_clause})"

    return (dict(data) for data in conn.execute(query))


def get_genotype_rowid(conn: sqlite3.Connection, variant_id: int, sample_id: int):
    conn.row_factory = sqlite3.Row
    return dict(
        conn.execute(
            f"SELECT genotypes.rowid FROM genotypes WHERE variant_id = {variant_id} AND sample_id = {sample_id}"
        ).fetchone()
    )["rowid"]


def update_sample(conn: sqlite3.Connection, sample: dict):
    if "id" not in sample:
        logging.debug("sample id is required")
        return

    sql_set = []
    sql_val = []

    for key, value in sample.items():
        if key != "id":
            sql_set.append(f"`{key}` = ? ")
            sql_val.append(value)

    query = "UPDATE samples SET " + ",".join(sql_set) + " WHERE id = " + str(sample["id"])
    conn.execute(query, sql_val)
    conn.commit()


def update_genotypes(conn: sqlite3.Connection, data: dict):
    if "variant_id" not in data and "sample_id" not in data:
        logging.debug("id is required")
        return

    sql_set = []
    sql_val = []

    for key, value in data.items():
        if key not in ("variant_id", "sample_id"):
            sql_set.append(f"`{key}` = ? ")
            sql_val.append(value)

    sample_id = data["sample_id"]
    variant_id = data["variant_id"]
    query = (
        "UPDATE genotypes SET "
        + ",".join(sql_set)
        + f" WHERE sample_id = {sample_id} AND variant_id = {variant_id}"
    )

    conn.execute(query, sql_val)
    conn.commit()


def create_triggers(conn):

    # variants count case/control on samples update
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS count_after_update_on_samples AFTER UPDATE ON samples
        WHEN new.phenotype <> old.phenotype
        BEGIN
            UPDATE variants
            SET 
                case_count_ref = case_count_ref + IIF( new.phenotype = 2 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=0) = 1, 1, 0 ) + IIF( old.phenotype = 2 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=0) = 1, -1, 0 ),
                
                case_count_het = case_count_het + IIF( new.phenotype = 2 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=1) = 1, 1, 0 ) + IIF( old.phenotype = 2 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=1) = 1, -1, 0 ),
                
                case_count_hom = case_count_hom + IIF( new.phenotype = 2 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=2) = 1, 1, 0 ) + IIF( old.phenotype = 2 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=2) = 1, -1, 0 ),

                control_count_ref = control_count_ref + IIF( new.phenotype = 1 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=0) = 1, 1, 0 ) + IIF( old.phenotype = 1 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=0) = 1, -1, 0 ),
                
                control_count_het = control_count_het + IIF( new.phenotype = 1 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=1) = 1, 1, 0 ) + IIF( old.phenotype = 1 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=1) = 1, -1, 0 ),
                
                control_count_hom = control_count_hom + IIF( new.phenotype = 1 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=2) = 1, 1, 0 ) + IIF( old.phenotype = 1 AND (SELECT count(shv.variant_id) FROM genotypes as shv WHERE sample_id=new.id AND variant_id=variants.id AND gt=2) = 1, -1, 0 )
                
            WHERE variants.id IN (SELECT shv2.variant_id FROM genotypes as shv2 WHERE shv2.sample_id=new.id) ;
        END;
        """
    )

    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS count_validation_positive_negative_after_update_on_genotypes AFTER UPDATE ON genotypes
        WHEN new.classification <> old.classification
        BEGIN
            UPDATE variants
            SET count_validation_positive = (SELECT count(shv.sample_id) FROM genotypes as shv WHERE shv.variant_id=new.variant_id AND shv.classification>0), 
                count_validation_negative = (SELECT count(shv.sample_id) FROM genotypes as shv WHERE shv.variant_id=new.variant_id AND shv.classification<0)
            WHERE id=new.variant_id;
        END;
        """
    )

    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS count_validation_positive_negative_variant_after_update_on_genotypes AFTER UPDATE ON genotypes
        WHEN new.classification <> old.classification
        BEGIN
            UPDATE samples
            SET count_validation_positive_variant = (SELECT count(shv.variant_id) FROM genotypes as shv WHERE shv.sample_id=new.sample_id AND shv.classification>0), 
                count_validation_negative_variant = (SELECT count(shv.variant_id) FROM genotypes as shv WHERE shv.sample_id=new.sample_id AND shv.classification<0)
            WHERE id = new.sample_id;
        END;
        """
    )


    tables_fields_triggered = {
        "variants": ["favorite", "classification", "tags", "comment"],
        "samples": [
            "classification",
            "tags",
            "comment",
            "family_id",
            "father_id",
            "mother_id",
            "sex",
            "phenotype",
        ],
        "genotypes": ["classification", "tags", "comment"],
    }

    for table in tables_fields_triggered:

        for field in tables_fields_triggered[table]:

            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS history_{table}_{field}
                AFTER UPDATE ON {table}
                WHEN old.{field} !=  new.{field}
                BEGIN
                    INSERT INTO history (
                        `user`,
                        `table`,
                        `table_rowid`,
                        `field`,
                        `before`,
                        `after`
                    )
                VALUES
                    (
                    current_user(),
                    "{table}",
                    new.rowid,
                    "{field}",
                    old.{field},
                    new.{field}
                    ) ;
                END;"""
            )


def create_database_schema(conn: sqlite3.Connection, fields: Iterable[dict] = None):

    if fields is None:
        fields = list(get_clean_fields())

    create_table_project(conn)
    create_table_metadatas(conn)
    create_table_fields(conn)

    variant_fields = (i for i in fields if i["category"] == "variants")
    create_table_variants(conn, variant_fields)

    ann_fields = (i for i in fields if i["category"] == "annotations")
    create_table_annotations(conn, ann_fields)

    sample_fields = (i for i in fields if i["category"] == "samples")
    create_table_samples(conn, sample_fields)

    create_table_selections(conn)




def import_reader(
    conn: sqlite3.Connection,
    reader: AbstractReader,
    pedfile: str = None,
    project:dict = None,
    import_id: str = None,
    ignored_fields: list = [],
    indexed_fields: list = [],
    progress_callback: Callable = None,
):

    tables = ["variants", "annotations", "genotypes"]
    fields = get_clean_fields(reader.get_fields())
    fields = get_accepted_fields(fields, ignored_fields)

    if not schema_exists(conn):
        LOGGER.debug("CREATE TABLE SCHEMA")
        create_database_schema(conn, fields)
    else:
        alter_table_from_fields(conn, fields)

    update_metadatas(conn, reader.get_metadatas())

    if project:
        update_project(conn, project)

    if progress_callback:
        progress_callback("Insert samples")
    if reader.filename:
        import_vcf = os.path.basename(reader.filename)
    else:
        import_vcf = None
    insert_samples(conn, samples=reader.get_samples(), import_id=import_id, import_vcf=import_vcf)


    insert_fields(conn, fields)


    if progress_callback:
        progress_callback("Insert variants. This can take a while")
    create_annotations_indexes(conn)
    insert_variants(
        conn,
        get_clean_variants(reader.get_variants()),
        total_variant_count=reader.number_lines,
        progress_callback=progress_callback,
        progress_every=1000,
    )

    if progress_callback:
        progress_callback("Indexation. This can take a while")

    vindex = {field["name"] for field in indexed_fields if field["category"] == "variants"}
    aindex = {field["name"] for field in indexed_fields if field["category"] == "annotations"}
    sindex = {field["name"] for field in indexed_fields if field["category"] == "samples"}

    try:
        create_indexes(conn, vindex, aindex, sindex, progress_callback=progress_callback)
    except:
        LOGGER.info("Index already exists")

    if progress_callback:
        progress_callback("Variants counts. This can take a while")
    update_variants_counts(conn, progress_callback)

    if progress_callback:
        progress_callback("Database creation complete")





