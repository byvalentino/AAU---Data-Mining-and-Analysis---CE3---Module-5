"""What every Module 5 lab needs: the unsolved marker, the stream, and the model.

Module 5 watches a model that is already in service. The model is trained by
`service/models.py` when you run `setup.sh`; the days it serves come from
`service/world.py`. Nothing here computes anything a lab is asked to compute --
it only fixes the grain, so that every lab and every check measures the same
thing and can print the choice beside the number.

The days are read from `data/stream.parquet`, which `setup.sh` generated once,
and are regenerated from `service/world.py` only when that file is absent. A lab
that generates its own data at import time is a lab whose numbers depend on when
it was run; a lab that reads a file `make check` has already verified is not.

One function is carried over rather than re-derived: `wilson_interval`, written
in Module 4, Lab 1. Block two buys truth by sampling, and a share measured on a
sample without an interval around it is a number pretending to be a fact.
"""
from __future__ import annotations

import logging
import math
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STREAM = HERE / "data" / "stream.parquet"

# ---- the grain, fixed here and printed beside anything computed from it -------

REFERENCE_DAYS = 10        # days 0-9: what the model trained on, and the monitor's reference
DAYS = 28                  # the whole watched period
WATCHED = "speed"          # the input the monitor tracks day by day

# The columns a monitor that watches more than one thing watches. Block one
# builds the monitor on WATCHED alone, because one column is enough to show
# what a yardstick is; block three points it at all four, because a graduate
# will point it at forty and every extra column buys another chance to fire
# on nothing. These are the model's own inputs -- watch what the model is
# fed, not a convenient subset of it.
WATCHED_COLUMNS = ["speed", "rssi1", "rssi2", "rssiC"]
SHIFT_THRESHOLD = 0.5      # reference standard deviations before anything is said
CONFIRMATIONS = 2          # days in a row over the line before anybody is woken
COOLDOWN = 5               # days of silence after a page, so one change pages once
SAMPLE_SIZE = 200          # rows hand-checked per day when truth is bought
Z_95 = 1.959963984540054

# The registry the released model is published through, and the alias that names
# the version in service. Module 3 chose these names; Module 5 keeps them, so a
# student who read Module 3's store finds the same three words here.
REGISTERED_MODEL = "aboard"
CHAMPION = "champion"
FEATURES = ["speed", "rssi1", "rssi2", "rssiC"]


class NotSolved(Exception):
    """A lab stub raises this. The check turns it into exit code 2.

    It is not an error. It means "you have not written this yet", which is a
    different state from "you wrote it and it is wrong", and the checks say so.
    """


class EnvironmentNotReady(Exception):
    """The tools or the data this module needs are missing. The checks exit 3.

    A third state, separate from the other two, because "this machine is not set
    up" is not "your code is wrong". A student told the second while the first is
    true goes hunting for a bug that was never there. Every check imports it from
    here, so the modules share one class rather than several with the same name.
    """


def load_lab(number: int):
    """Import another lab by its number, so one lab can build on the last.

    Lab 2 measures its input level with Lab 1's input_shift, and Lab 3 alerts on
    Lab 1's shift series. Lab 4 uses neither: it works on the registry and the
    service, so it can be written in any order.

    Importing by number rather than by module name means the same line works
    whether the file holds your own attempt or the shipped solution.

        from lab_support import load_lab
        shift = load_lab(1).input_shift
    """
    import importlib.util

    matches = sorted((HERE / "labs").glob(f"{number:02d}_*.py"))
    if not matches:
        raise FileNotFoundError(f"no lab {number:02d} in {HERE / 'labs'}")
    specification = importlib.util.spec_from_file_location(f"lab{number:02d}", matches[0])
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# ---- the world under watch ---------------------------------------------------

def _stream() -> pd.DataFrame:
    """Every generated day, from the file setup.sh wrote, or from the generator.

    Read once and kept, because every lab asks for several days and reading the
    same Parquet file eight times is eight times the work for the same rows.
    """
    global _STREAM_CACHE
    if _STREAM_CACHE is None:
        if STREAM.exists():
            _STREAM_CACHE = pd.read_parquet(STREAM)
        else:
            logging.getLogger(__name__).warning(
                "data/stream.parquet is missing, so the days are being generated "
                "in this process instead -- run  bash setup.sh  to write it once")
            from service.world import stream
            _STREAM_CACHE = stream(DAYS)
    return _STREAM_CACHE


_STREAM_CACHE = None


def reference_frame() -> pd.DataFrame:
    """Days 0 to 9 — what the model was trained on, and what 'normal' means."""
    frame = _stream()
    return frame[frame["day"] < REFERENCE_DAYS].reset_index(drop=True)


def day_frame(number: int) -> pd.DataFrame:
    """One served day, features and truth. Truth is here only so a lab can buy it."""
    frame = _stream()
    return frame[frame["day"] == number].reset_index(drop=True)


def approved_model():
    """The model the registry currently points at. Trained by setup.sh."""
    from service import models
    if not (models.ARTEFACTS / "registry.json").exists():
        models.build()
    if not (models.ARTEFACTS / "registry.json").exists():
        raise EnvironmentNotReady(
            "service/artefacts/registry.json is still missing after training "
            "the model")
    version = models.load_registry()["approved"]
    artefact = models.load(version)
    return artefact["model"], artefact["features"]


def load_champion():
    """The released model, loaded from the platform registry through its alias.

    `models:/aboard@champion` is a lookup, not a path: it names whichever version
    the alias points at right now. The object that comes back enforces the
    signature recorded when the run was logged -- it reorders columns by name and
    refuses a frame whose columns are renamed or missing.
    """
    from service import models
    try:
        return models.load_champion()
    except Exception as unready:      # no store, or nothing released in it yet
        raise EnvironmentNotReady(
            f"the registered model {REGISTERED_MODEL!r} has no {CHAMPION!r} to "
            f"load ({unready}); run  bash setup.sh  to build the store") from None


def predictions(model, frame: pd.DataFrame, features=None):
    """What the model says about a frame, with absent beacons filled the same way
    they were filled in training. Training and serving must fill alike, or the
    model is asked a question in a language it was never taught -- Module 3."""
    from service import models
    return model.predict(models.prepare(frame, features or models.FEATURES))


# ---- carried over from Module 4, Lab 1 ---------------------------------------

def wilson_interval(successes: int, trials: int, z: float = Z_95):
    """A 95 per cent interval for a share measured on a sample (Wilson, 1927).

    Written in Module 4, Lab 1, and imported here rather than rewritten. It is
    used on bought truth: hand-check 200 rows, get 178 right, and the honest
    statement is not "0.890" but the interval this returns.
    """
    if trials == 0:
        return (0.0, 1.0)
    centre = (successes + z * z / 2) / (trials + z * z)
    half = z / (trials + z * z) * math.sqrt(
        successes * (trials - successes) / trials + z * z / 4)
    return (max(0.0, centre - half), min(1.0, centre + half))
