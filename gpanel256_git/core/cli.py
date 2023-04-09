import argparse
import os
import sys
from functools import partial

import progressbar
from columnar import columnar
from gpanel256.core import sql, vql, command
from gpanel256.core.readerfactory import create_reader
from gpanel256.core.querybuilder import *
from gpanel256 import LOGGER


def display_sql_results(data, headers, *args, **kwargs):
    print(
        columnar(
            data,
            headers=headers,
            no_borders=True,
            **kwargs,
        )
    )


def create_db(args):
    if not args.db:

        args.db = args.input + ".db"

    conn = sql.get_sql_connection(args.db)
    if conn:

        with create_reader(args.input) as reader:
            sql.import_reader(conn, reader, import_id = args.import_id)

        print("Successfully created database!")

def show(args, conn):
    if args.table == "fields":
        display_sql_results(
            (i.values() for i in sql.get_fields(conn)),
            ["id", "name", "table", "type", "description"],
        )

    if args.table == "samples":
        display_sql_results((i.values() for i in sql.get_samples(conn)), ["id", "name"])

    if args.table == "selections":
        display_sql_results(
            (i.values() for i in sql.get_selections(conn)),
            ["id", "name", "variant_count"],
        )

    if args.table == "wordsets":
        display_sql_results((i.values() for i in sql.get_wordsets(conn)), ["id", "word_count"])


def remove(args, conn):
    for name in args.names:
        rows_removed = sql.delete_selection_by_name(conn, name)
        if rows_removed:
            print(f"Successfully removed {rows_removed} variants from selection {name}")
        else:
            print(f"Could not remove selection {name}")
    return 0


def select(args, conn):
    query = "".join(args.vql)
    vql_command = None

    try:
        cmd = vql.parse_one_vql(query)
    except (vql.textx.TextXSyntaxError, vql.VQLSyntaxError) as e:
        print("%s: %s, col: %d" % (e.__class__.__name__, e.message, e.col))
        print("For query:", query)
        return 1

    if cmd["cmd"] == "select_cmd" and args.to_selection:
        vql_command = partial(
            command.create_cmd,
            conn,
            args.to_selection,
            source=cmd["source"],
            filters=cmd["filters"],
        )

    try:


        if not isinstance(ret, dict):
            ret = list(ret)
    except (sqlite3.DatabaseError, vql.VQLSyntaxError) as e:
        LOGGER.exception(e)
        return 1

    LOGGER.debug("SQL result: %s", ret)

    if cmd["cmd"] in ("select_cmd",) and not args.to_selection:
        display_sql_results((i.values() for i in ret), ["id"] + cmd["fields"])

    if (
        cmd["cmd"] in ("drop_cmd", "import_cmd", "create_cmd", "set_cmd", "bed_cmd")
        or args.to_selection
    ):
        display_query_status(ret)

    return 0


def main():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog),
        description="""
gpanel256 cli mode .\n
The env variable $gpanel256_DB """,
        epilog="""Examples:

    $ gpanel256-cli show --db my_database.db samples
    or
    $ export gpanel256_DB=my_database.db
    $ gpanel256-cli show samples""",
    )
    parser.add_argument(
        "-vv",
        "--verbose",
        nargs="?",
        default="error",
        choices=["debug", "info", "critical", "error", "warning"],
    )

    sub_parser = parser.add_subparsers(dest="subparser")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--db", help="SQLite database. By default, $gpanel256_DB is used.")

    createdb_parser = sub_parser.add_parser(
        "createdb",
        help="Build a SQLite database from a vcf file",
        parents=[parent_parser],
        epilog=""" :

        $ gpanel256-cli "
        """,
    )
    createdb_parser.add_argument("-i", "--input", help="VCF file path", required=True)

    createdb_parser.add_argument(
        "-m", "--import_id", help="Import ID to create a tag for each samples (optional, default <DATE>)."
    )
    createdb_parser.set_defaults(func=create_db)


    show_parser = sub_parser.add_parser(
        "show", help="Display table content", parents=[parent_parser]
    )
    show_parser.add_argument(
        "table",
        choices=["fields", "selections", "samples", "wordsets"],
        help="Possible names of tables.",
    )
    show_parser.set_defaults(func=show)

    remove_parser = sub_parser.add_parser(
        "remove", help="remove selection", parents=[parent_parser]
    )
    remove_parser.add_argument("names", nargs="+", help="Name(s) of selection(s).")
    remove_parser.set_defaults(func=remove)

    select_parser.add_argument(
        "-l",
        "--limit",
        help="Limit the number of lines in output.",
        type=int,
        default=100,
    )

    select_parser.add_argument(
        "-s", "--to-selection", help="Save SELECT query into a selection name."
    )
    select_parser.set_defaults(func=select)


    if "html" in sys.argv:
        return parser
    args = parser.parse_args()

    LOGGER.setLevel(args.verbose.upper())
    
    if "gpanel256_DB" in os.environ:
        args.db = os.environ["gpanel256_DB"]

    elif "db" not in dir(args) and args.subparser != "createdb":
        print("You must specify a database file via $gpanel256_DB or --db argument")
        print("Use --help for more information")
        return 1

    if args.db and args.subparser != "createdb":
        conn = sql.get_sql_connection(args.db)
        return args.func(args, conn)
    if args.subparser == "createdb":
        return create_db(args)

    print(
        "You specified no database to open, asked for none to be created, there is nothing more I can do!"
    )
    return 1


if __name__ == "__main__":
    exit(main())
