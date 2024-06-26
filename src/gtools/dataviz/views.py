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
from sdsstools.utils import Timer

from gtools import log


if TYPE_CHECKING:
    pass


def create_sdss_id_to_catalog_view(
    view_name: str = "sdss_id_to_catalog",
    schema: str = "catalogdb",
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
        f"SELECT * FROM pg_matviews WHERE matviewname = '{view_name}';"
    )
    view_exists = view_query.fetchone() is not None

    if view_exists:
        if drop_existing:
            log.warning(f'Droping existing view "{view_name}"')
            database.execute_sql(
                f"DROP MATERIALIZED VIEW IF EXISTS catalogdb.{view_name};"
            )
        else:
            raise ValueError(f'View "{view_name}" already exists.')

    # We build the query manually so that the resulting query is easy to read in
    # the materialized view.
    tables = database.get_tables(schema="catalogdb")
    catalog_to_tables = [table for table in tables if table.startswith("catalog_to_")]

    select_columns_list: list[str] = []
    aliases: list[str] = []
    query = """
    CREATE MATERIALIZED VIEW {schema}.{view_name} TABLESPACE pg_default AS
    SELECT row_number() OVER () as pk,
           catalogdb.sdss_id_flat.sdss_id,
           catalogdb.catalog.catalogid,
           catalogdb.catalog.version_id,
           catalogdb.catalog.lead,
{select_columns}
        FROM catalogdb.sdss_id_flat
        JOIN catalogdb.catalog
            ON sdss_id_flat.catalogid = catalog.catalogid
    """

    for c2table in sorted(catalog_to_tables):
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
        aliases.append(alias)

        # Gaia DR2 and 2MASS PSC are a special case because in v0.1 and v0.5 we
        # did not explicitely cross-match them as they were complete in TIC v8. But
        # in v1 we are not using the TIC and we did cross-match them using Gaia DR3
        # best neighbour tables. Here we need to include either one of the values
        # (only one of them will be non-NULL) in the view.

        if table == "gaia_dr2_source":
            select_columns_list.append(
                "COALESCE(catalogdb.gaia_dr2_source.source_id, "
                f"catalogdb.tic_v8.gaia_int) AS {alias}"
            )
        elif table == "twomass_psc":
            select_columns_list.append(
                f"COALESCE(catalogdb.twomass_psc.pts_key, tm2.pts_key) AS {alias}"
            )
        else:
            select_columns_list.append(f"catalogdb.{table}.{pk} AS {alias}")

        query += f"""
        LEFT JOIN catalogdb.{c2table}
            ON catalog.catalogid = {c2table}.catalogid
            AND {c2table}.best
            AND {c2table}.version_id = catalog.version_id
        LEFT JOIN catalogdb.{table}
            ON catalogdb.{c2table}.target_id = catalogdb.{table}.{pk}
        """

        if table == "twomass_psc":
            # catalog_to_tic_v8 has already been added (its alphabetically before
            # catalog_to_twomass_psc) so we only add the extra join here. We need to
            # use an alias since there is already a direct join
            # from catalog_to_twomass_psc to twomass_psc.
            query += """
        LEFT JOIN catalogdb.twomass_psc AS tm2
            ON catalogdb.tic_v8.twomass_psc = tm2.designation
        """

    select_columns: str = ""
    for column in select_columns_list:
        comma = "," if column != select_columns_list[-1] else ""
        select_columns += f"           {column}{comma}\n"

    query = textwrap.dedent(
        query.format(
            select_columns=select_columns,
            schema=schema,
            view_name=view_name,
        )
    )

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
