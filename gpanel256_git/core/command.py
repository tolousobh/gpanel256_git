
import sqlite3
import os
import functools


from gpanel256.core.querybuilder import build_sql_query
from gpanel256.core import sql, vql

from gpanel256.core.reader import BedReader

from gpanel256 import LOGGER


def select_cmd(
    conn: sqlite3.Connection,
    fields={"variants": ["chr", "pos", "ref", "alt"]},
    source="variants",
    filters={},
    order_by=None,
    order_desc=True,
    group_by=[],
    having={},  # {"op":">", "value": 3  }
    limit=50,
    offset=0,
    **kwargs,
):

    query = build_sql_query(
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
    LOGGER.debug("command:select_cmd:: %s", query)
    for i in conn.execute(query):

        yield {k.replace("(", "[").replace(")", "]"): v for k, v in dict(i).items()}


def count_cmd(
    conn: sqlite3.Connection,
    fields=["chr", "pos", "ref", "alt"],
    source="variants",
    filters={},
    group_by=[],
    having={},
    **kwargs,
):



    variants_fields = set(field["name"] for field in sql.get_field_by_category(conn, "variants"))

    if set(fields).issubset(variants_fields) and not filters and not group_by:

        LOGGER.debug("command:count_cmd:: cached from selections table")
        return {
            "count": conn.execute(
                f"SELECT count FROM selections WHERE name = '{source}'"
            ).fetchone()[0]
        }

    query = build_sql_query(
        conn,
        fields=fields,
        source=source,
        filters=filters,
        limit=None,
        offset=None,
        order_by=None,
        group_by=group_by,
        having=having,
        **kwargs,
    )


    LOGGER.debug("command:count_cmd:: %s", query)
    return {"count": sql.count_query(conn, query)}


def drop_cmd(conn: sqlite3.Connection, feature: str, name: str, **kwargs):

    accept_features = ("selections", "wordsets")

    feature = feature.lower()

    if feature not in accept_features:
        raise vql.VQLSyntaxError(f"{feature} doesn't exists")

    affected_lines = 0
    if feature == "selections":
        affected_lines = sql.delete_selection_by_name(conn, name)

    if feature == "wordsets":
        affected_lines = sql.delete_wordset_by_name(conn, name)

    return {"success": (affected_lines > 0)}


def create_cmd(
    conn: sqlite3.Connection,
    target: str,
    source="variants",
    filters=dict(),
    count=None,
    **kwargs,
):

    if target is None:
        return {}

    sql_query = build_sql_query(
        conn,
        fields=["id"],
        source=source,
        filters=filters,
        limit=None,
        **kwargs,
    )

    LOGGER.debug("command:create_cmd:: %s", sql_query)
    selection_id = sql.insert_selection_from_source(conn, target, source, filters, count)
    return dict() if selection_id is None else {"id": selection_id}


def set_cmd(conn: sqlite3.Connection, target: str, first: str, second: str, operator, **kwargs):

    if target is None or first is None or second is None or operator is None:
        return {}

    query_first = build_sql_query(conn, ["id"], first, limit=None)
    query_second = build_sql_query(conn, ["id"], second, limit=None)

    func_query = {
        "|": sql.union_variants,
        "-": sql.subtract_variants,
        "&": sql.intersect_variants,
    }

    sql_query = func_query[operator](query_first, query_second)
    LOGGER.debug("command:set_cmd:: %s", sql_query)

    selection_id = sql.insert_selection_from_sql(conn, sql_query, target, from_selection=False)
    return dict() if selection_id is None else {"id": selection_id}


def show_cmd(conn: sqlite3.Connection, feature: str, **kwargs):

    accepted_features = {
        "selections": sql.get_selections,
        "fields": sql.get_fields,
        "samples": sql.get_samples,
        "wordsets": sql.get_wordsets,
    }

    feature = feature.lower()

    if feature not in accepted_features:
        raise vql.VQLSyntaxError(f"option {feature} doesn't exists")

    for item in accepted_features[feature](conn):
        yield item


def execute_all(conn: sqlite3.Connection, sql_source: str):

    for sql_obj in vql.parse_sql(vql_source):
        cmd = create_command_from_obj(conn, sql_obj)
        yield cmd()

