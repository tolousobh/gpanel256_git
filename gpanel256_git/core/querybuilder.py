
import sqlite3
import re
from functools import lru_cache

from gpanel256.core import sql

import gpanel256.constants as cst


from gpanel256 import LOGGER


WORDSET_FUNC_NAME = "WORDSET"

PY_TO_SQL_OPERATORS = {
    "$eq": "=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$in": "IN",
    "$nin": "NOT IN",
    "$ne": "!=",
    "$regex": "REGEXP",
    "$nregex": "NOT REGEXP",
    "$and": "AND",
    "$or": "OR",
    "$has": "HAS",
    "$nhas": "NOT HAS",
}



def filters_to_flat(filters: dict):


    flatten = []
    for k, v in filters.items():
        if isinstance(v, list):
            for i in v:
                flatten += filters_to_flat(i)

        else:
            if filters not in flatten:
                flatten.append(filters)

    return flatten


def is_annotation_join_required(fields, filters, order_by=None) -> bool:


    for field in fields:
        if field.startswith("ann."):
            return True

    if order_by:
        for by in order_by:
            field, direction = by
            if field.startswith("ann."):
                return True

    for condition in filters_to_flat(filters):

        condition = list(condition.keys())[0]
        if condition.startswith("ann."):
            return True

    return False


def samples_join_required(fields, filters, order_by=None) -> list:

    samples = set()

    for field in fields:
        if field.startswith("samples"):
            _, *sample, _ = field.split(".")
            sample = ".".join(sample)
            samples.add(sample)

    if order_by:
        for by in order_by:
            field, direction = by
            if field.startswith("samples"):
                _, *sample, _ = field.split(".")
                sample = ".".join(sample)
                samples.add(sample)

    for condition in filters_to_flat(filters):
        key = list(condition.keys())[0]
        if key.startswith("samples"):
            _, *sample, _ = key.split(".")
            sample = ".".join(sample)
            samples.add(sample)

    return list(samples)




def fields_to_sql(fields, use_as=False) -> list:

    sql_fields = []

    for field in fields:

        if field.startswith("ann."):
            sql_field = f"`annotations`.`{field[4:]}`"
            if use_as:
                sql_field = f"{sql_field} AS `ann.{field[4:]}`"
            sql_fields.append(sql_field)

        elif field.startswith("samples."):

            _, *name, value = field.split(".")

            name = ".".join(name)

            sql_field = f"`sample_{name}`.`{value}`"
            if use_as:
                sql_field = f"{sql_field} AS `samples.{name}.{value}`"
            sql_fields.append(sql_field)

        else:
            sql_fields.append(f"`variants`.`{field}`")

    return sql_fields


def condition_to_sql(item: dict, samples=None) -> str:


    k = list(item.keys())[0]
    v = item[k]

    if k.startswith("ann."):
        table = "annotations"
        k = k[4:]

    elif k.startswith("samples."):
        table = "samples"
        _, *name, k = k.split(".")
        name = ".".join(name)

    else:
        table = "variants"

    field = f"`{table}`.`{k}`"

    if isinstance(v, dict):
        vk, vv = list(v.items())[0]
        operator = vk
        value = vv
    else:
        operator = "$eq"
        value = v

    if isinstance(value, str):
        value = value.replace("'", "''")

    sql_operator = PY_TO_SQL_OPERATORS[operator]


    if "REGEXP" in sql_operator:
        special_caracter = "[]+.?*()^$"
        if not set(str(value)) & set(special_caracter):
            sql_operator = "LIKE" if sql_operator == "REGEXP" else "NOT LIKE"
            value = f"%{value}%"

    if "HAS" in sql_operator:
        field = f"'{cst.HAS_OPERATOR}' || {field} || '{cst.HAS_OPERATOR}'"
        sql_operator = "LIKE" if sql_operator == "HAS" else "NOT LIKE"
        value = f"%{cst.HAS_OPERATOR}{value}{cst.HAS_OPERATOR}%"

    if isinstance(value, str):
        value = f"'{value}'"

    if isinstance(value, bool):
        value = int(value)

    if value is None:
        if operator == "$eq":
            sql_operator = "IS"

        if operator == "$ne":
            sql_operator = "IS NOT"

        value = "NULL"

    if isinstance(value, dict):
        if "$wordset" in value:
            wordset_name = value["$wordset"]
            value = f"(SELECT value FROM wordsets WHERE name = '{wordset_name}')"

    if isinstance(value, list) or isinstance(value, tuple):
        value = "(" + ",".join([f"'{i}'" if isinstance(i, str) else f"{i}" for i in value]) + ")"

    operator = None
    condition = ""

    if table == "samples":

        if name == "$any":
            operator = "OR"

        if name == "$all":
            operator = "AND"

        if operator and samples:

            condition = (
                "("
                + f" {operator} ".join(
                    [f"`sample_{sample}`.`{k}` {sql_operator} {value}" for sample in samples]
                )
                + ")"
            )

        else:
            condition = f"`sample_{name}`.`{k}` {sql_operator} {value}"

    else:
        condition = f"{field} {sql_operator} {value}"

    return condition


def remove_field_in_filter(filters: dict, field: str = None) -> dict:

    def recursive(obj):

        output = {}
        for k, v in obj.items():
            if k in ["$and", "$or"]:
                temp = []
                for item in v:
                    rec = recursive(item)
                    if field not in item and rec:
                        temp.append(rec)
                if temp:
                    output[k] = temp
                    return output

            else:
                output[k] = v
                return output

    return recursive(filters) or {}


def filters_to_sql(filters: dict, samples=None) -> str:

    def recursive(obj):

        conditions = ""
        for k, v in obj.items():
            if k in ["$and", "$or"]:
                conditions += (
                    "(" + f" {PY_TO_SQL_OPERATORS[k]} ".join([recursive(item) for item in v]) + ")"
                )

            else:
                conditions += condition_to_sql(obj, samples)

        return conditions

    query = recursive(filters)


    return query



def build_sql_query(
    conn: sqlite3.Connection,
    fields,
    source="variants",
    filters={},
    order_by=[],
    limit=50,
    offset=0,
    selected_samples=[],
    **kwargs,
):


    samples_ids = {i["name"]: i["id"] for i in sql.get_samples(conn)}

    sql_fields = ["`variants`.`id`"] + fields_to_sql(fields, use_as=True)

    sql_query = f"SELECT DISTINCT {','.join(sql_fields)} "

    sql_query += "FROM variants"

    if is_annotation_join_required(fields, filters, order_by):
        sql_query += " LEFT JOIN annotations ON annotations.variant_id = variants.id"

    if source != "variants":
        sql_query += (
            " INNER JOIN selection_has_variant sv ON sv.variant_id = variants.id "
            f"INNER JOIN selections s ON s.id = sv.selection_id AND s.name = '{source}'"
        )

    filters_fields = " ".join([list(i.keys())[0] for i in filters_to_flat(filters)])

    if "$all" in filters_fields or "$any" in filters_fields:
        join_samples = list(samples_ids.keys())

    else:
        join_samples = samples_join_required(fields, filters, order_by)

    for sample_name in join_samples:
        if sample_name in samples_ids:
            sample_id = samples_ids[sample_name]
            sql_query += f""" LEFT JOIN genotypes `sample_{sample_name}` ON `sample_{sample_name}`.variant_id = variants.id AND `sample_{sample_name}`.sample_id = {sample_id}"""

    if filters:
        where_clause = filters_to_sql(filters, join_samples)
        if where_clause and where_clause != "()":
            sql_query += " WHERE " + where_clause

    if order_by:

        order_by_clause = []
        for item in order_by:
            field, direction = item

            field = fields_to_sql([field])[0]

            direction = "ASC" if direction else "DESC"
            order_by_clause.append(f"{field} {direction}")

        order_by_clause = ",".join(order_by_clause)

        sql_query += f" ORDER BY {order_by_clause}"

    if limit:
        sql_query += f" LIMIT {limit} OFFSET {offset}"

    return sql_query


def build_vql_query(
    fields,
    source="variants",
    filters={},
    order_by=[],
    **kwargs,
):

    select_clause = ",".join(fields_to_vql(fields))

    where_clause = filters_to_vql(filters)

    if where_clause and where_clause != "()":
        where_clause = f" WHERE {where_clause}"
    else:
        where_clause = ""

    order_by_clause = ""
    if order_by:
        order_by_clause = []
        for item in order_by:
            field, direction = item
            field = fields_to_vql([field])[0]
            direction = "ASC" if direction else "DESC"
            order_by_clause.append(f"{field} {direction}")

        order_by_clause = " ORDER BY " + ",".join(order_by_clause)

    return f"SELECT {select_clause} FROM {source}{where_clause}{order_by_clause}"
