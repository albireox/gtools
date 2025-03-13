#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-13
# @Filename: monitor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import time
from threading import Lock

from typing import Annotated

import numpy
import seaborn
import typer
from astropy.io import fits
from matplotlib import pyplot as plt
from watchdog.events import (
    FileCreatedEvent,
    FileMovedEvent,
    PatternMatchingEventHandler,
)
from watchdog.observers.polling import PollingObserver

from sdsstools import get_logger

from gtools.lvm.pipe import extract_and_stitch, plot_ifu_data


log = get_logger("gtools-lvm-monitor", use_rich_handler=True)

seaborn.set_theme(style="white")
plt.ioff()


def process_files(files: list[pathlib.Path], outpath: pathlib.Path):
    """Processes a set of spectrograph raw files."""

    files = [pathlib.Path(file).absolute() for file in files]
    outpath = pathlib.Path(outpath).absolute()

    exp_no = get_spec_frameno(files[0])

    log.debug("Running quick extraction with mode='sum'.")
    data_sum = extract_and_stitch(files, detrend=True, mode="sum")

    log.debug("Running quick extraction with mode='cent'.")
    data_cent = extract_and_stitch(files, detrend=True, mode="cent")

    log.debug("Creating IFU image.")
    plot_ifu_data(data_sum, filename=outpath / f"ifu-{exp_no:08d}.pdf")

    log.debug("Plotting histogram data.")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    bins = numpy.arange(0, 2**16, 5000)
    nw = 4086
    perc = 80

    count_global, bins_global = numpy.histogram(data_cent, bins)
    axes[0, 0].hist(count_global, bins=bins_global, color="k")
    axes[0, 0].set_xlabel("Counts")
    axes[0, 0].set_title("Full wavelength range")

    perc_global = numpy.percentile(data_cent, perc)
    log.debug(f"Percentile {perc}% (global): {perc_global:.2f}.")

    count_b, bins_b = numpy.histogram(data_cent[:, 0:nw], bins)
    axes[0, 1].hist(count_b, bins=bins_b, color="b")
    axes[0, 1].set_xlabel("Counts")
    axes[0, 1].set_title("Blue")

    perc_b = numpy.percentile(data_cent[:, 0:nw], perc)
    log.debug(f"Percentile {perc}% (b): {perc_b:.2f}.")

    count_r, bins_r = numpy.histogram(data_cent[:, nw : 2 * nw], bins)
    axes[1, 0].hist(count_b, bins=bins_r, color="r")
    axes[1, 0].set_xlabel("Counts")
    axes[1, 0].set_title("Red")

    perc_r = numpy.percentile(data_cent[:, nw : 2 * nw], perc)
    log.debug(f"Percentile {perc}% (r): {perc_r:.2f}.")

    count_z, bins_z = numpy.histogram(data_cent[:, 2 * nw :], bins)
    axes[1, 1].hist(count_z, bins=bins_z, color="m")
    axes[1, 1].set_xlabel("Counts")
    axes[1, 1].set_title("IR")

    perc_z = numpy.percentile(data_cent[:, 2 * nw :], perc)
    log.debug(f"Percentile {perc}% (z): {perc_z:.2f}.")

    fig.tight_layout()
    fig.savefig(outpath / f"hist-{exp_no:08d}.pdf")


class SpecPatternEventHandler(PatternMatchingEventHandler):
    """Handles newly created spectrograph files."""

    def __init__(self, outpath: pathlib.Path):
        self.lock = Lock()
        self.outpath = outpath

        super().__init__(
            patterns=["*-b1-*.fits.gz"],
            ignore_directories=True,
            case_sensitive=True,
        )

    def on_any_event(self, event: FileCreatedEvent | FileMovedEvent):
        """Runs the co-add code when a new file is created/moved."""

        # Do not process more than one file at the same time.
        while self.lock.locked():
            time.sleep(1)

        self.lock.acquire()

        try:
            if event.event_type == "moved":
                new_file = event.dest_path
            elif event.event_type == "created":
                new_file = event.src_path
            else:
                log.debug(f"Not handling event {event!r}")
                return

            if isinstance(new_file, bytes):
                new_file = new_file.decode("utf-8")

            assert isinstance(new_file, str), f"new_file is not a string: {new_file!r}"

            if new_file is None or new_file == "":
                return

            path = pathlib.Path(new_file).absolute()

            if not path.exists():
                log.warning(f"Detected file {path!s} does not exist!")
                return

            image_type = fits.getval(str(path), "IMAGETYP")
            if image_type != "object":
                log.info(f"Detected file {path}. Not an object image, skipping.")
                return

            frame_no = get_spec_frameno(path)
            log.info(f"Processing spectrograph frame {frame_no}.")

            elapsed: float = 0
            while True:
                files = list(path.parents[0].glob(path.name.replace("-b1-", "-*-")))
                if len(files) != 9:
                    if elapsed >= 15:
                        log.error(
                            "Not all spectrograph camera files were found. "
                            f"Ignoring image {path!s}."
                        )
                        return
                    time.sleep(1)
                    elapsed += 1
                break

            process_files(files, self.outpath)

            log.info(f"Finished processing spectrograph frame {frame_no}.")

        except Exception as err:
            log.error(f"Error found while processing file: {err}")

        finally:
            # Release the lock so other files can be processed.
            self.lock.release()


def monitor_spectro(path: str | pathlib.Path, outpath: str | pathlib.Path):
    """Monitors and reduces new spectro files."""

    path = pathlib.Path(path).absolute()

    outpath = pathlib.Path(outpath).absolute()
    outpath.mkdir(exist_ok=True, parents=True)

    observer = PollingObserver(timeout=5)
    handler = SpecPatternEventHandler(outpath=outpath)

    observer.schedule(handler, str(path))
    observer.start()

    log.info(f"Watching directory {path!s}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    except Exception as err:
        log.exception(f"Error found while monitoring {path!s}: {err}", exc_info=err)

    observer.join()


def get_spec_frameno(file: pathlib.Path):
    """Returns the spectrograph sequence number for a spectrograph file."""

    return int(file.name.split("-")[-1].split(".")[0])


cli = typer.Typer(
    rich_markup_mode="rich",
    context_settings={"obj": {}},
    no_args_is_help=True,
    help="Monitor LVM spectro files.",
)


@cli.command()
def monitor_cli(
    spectro_path: Annotated[
        pathlib.Path,
        typer.Argument(
            file_okay=False,
            dir_okay=True,
            exists=True,
            help="The path to the spectro directory to monitor.",
        ),
    ],
    outpath: Annotated[
        pathlib.Path,
        typer.Argument(
            file_okay=False,
            dir_okay=True,
            exists=False,
            help="The path to save the output files.",
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            help="Show debug messages.",
        ),
    ] = False,
):
    """Monitors and reduces new spectro files."""

    if verbose:
        log.set_level(5)
        log.sh.setLevel(5)

    monitor_spectro(spectro_path, outpath)


if __name__ == "__main__":
    cli()
