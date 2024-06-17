#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-05-09
# @Filename: views.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import textwrap

from typing import TYPE_CHECKING

from sdssdb.peewee.sdss5db import database
from sdsstools import get_logger
from sdsstools.utils import Timer


if TYPE_CHECKING:
    pass


log = get_logger("gtools.datavix.views", use_rich_handler=True)


def create_sdss_id_to_catalog_view(
    drop_existing: bool = False,
    local: bool = False,
    show_query=False,
):
    """Creates a view that maps SDSS IDs to parent catalogue PKs."""

    if local:
        database.connect(user="u0931042", host="localhost", port=7602)
    else:
        database.connect(user="u0931042", host="pipelines.sdss.org", port=5432)

    assert database.connected, "Database not connected."

    view_query = database.execute_sql(
        "SELECT * FROM pg_matviews WHERE matviewname = 'sdss_id_to_catalog';"
    )
    view_exists = view_query.fetchone() is not None
    print(view_exists)

    if view_exists:
        if drop_existing:
            log.warning('Droping existing view "sdss_id_to_catalog"')
            database.execute_sql(
                "DROP MATERIALIZED VIEW IF EXISTS catalogdb.sdss_id_to_catalog;"
            )
        else:
            raise ValueError('View "sdss_id_to_catalog" already exists.')

    # We build the query manually so that the resulting query is easy to read in
    # the materialized view.
    tables = database.get_tables(schema="catalogdb")
    catalog_to_tables = [table for table in tables if table.startswith("catalog_to_")]

    select_columns_list: list[str] = []
    aliases: list[str] = []
    query = """
    CREATE MATERIALIZED VIEW catalogdb.sdss_id_to_catalog TABLESPACE pg_default AS
    SELECT row_number() OVER () as pk,
           catalogdb.sdss_id_flat.sdss_id,
           catalogdb.catalog.catalogid,
           catalogdb.catalog.version_id,
{select_columns}
        FROM catalogdb.sdss_id_flat
        JOIN catalogdb.catalog
            ON sdss_id_flat.catalogid = catalog.catalogid
    """

    for c2table in catalog_to_tables:
        table = c2table.replace("catalog_to_", "")

        if c2table in ["catalog_to_sdss_dr13_photoobj"]:
            continue
        if table in ["skies_v1", "skies_v2"]:
            continue
        if not database.table_exists(table, schema="catalogdb"):
            continue

        pks = database.get_primary_keys(table, schema="catalogdb")

        if c2table == "catalog_to_sdss_dr13_photoobj_primary":
            table = "sdss_dr13_photoobj"
            pks = ["objid"]

        if len(pks) != 1:
            log.warning(f"Skipping table {table!r} with multiple primary keys.")
            continue

        pk = pks[0]
        alias = f"{table}__{pk}"
        select_columns_list.append(f"catalogdb.{table}.{pk} AS {alias}")
        aliases.append(alias)

        query += f"""
        LEFT JOIN catalogdb.{c2table}
            ON catalog.catalogid = {c2table}.catalogid
            AND {c2table}.best
            AND {c2table}.version_id = catalog.version_id
        LEFT JOIN catalogdb.{table}
            ON catalogdb.{c2table}.target_id = catalogdb.{table}.{pk}
        """

    select_columns: str = ""
    for column in select_columns_list:
        comma = "," if column != select_columns_list[-1] else ""
        select_columns += f"           {column}{comma}\n"

    query = textwrap.dedent(query.format(select_columns=select_columns))

    if show_query:
        log.info("The following query will be run:")
        log.info(query)

    log.info("Creating view 'sdss_id_to_catalog' ...")

    with Timer() as timer:
        with database.atomic():
            database.execute_sql("SET LOCAL search_path TO catalogdb;")
            database.execute_sql("SET LOCAL max_parallel_workers = 64;")
            database.execute_sql("SET LOCAL max_parallel_workers_per_gather = 32;")
            database.execute_sql("SET LOCAL effective_io_concurrency = 500;")
            database.execute_sql('SET LOCAL effective_cache_size = "1TB";')
            database.execute_sql('SET LOCAL work_mem = "1000MB";')
            database.execute_sql('SET LOCAL temp_buffers = "1000MB";')

            database.execute_sql(query)

    log.debug(f"Query executed in {timer.elapsed:.2f} seconds.")

    log.info("Creating indices ..")

    database.execute_sql("CREATE INDEX ON sdss_id_to_catalog (pk);")
    database.execute_sql("CREATE INDEX ON sdss_id_to_catalog (sdss_id);")
    database.execute_sql("CREATE INDEX ON sdss_id_to_catalog (catalogid);")
    database.execute_sql("CREATE INDEX ON sdss_id_to_catalog (version_id);")

    for alias in aliases:
        database.execute_sql(f"CREATE INDEX ON sdss_id_to_catalog ({alias});")

    log.info("Running VACUUM ANALYZE ...")
    database.execute_sql("VACUUM ANALYZE sdss_id_to_catalog;")

    log.info("Done.")
