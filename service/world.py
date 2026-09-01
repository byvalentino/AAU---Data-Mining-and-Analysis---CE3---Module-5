#!/usr/bin/env python3
"""A stream of days for the monitor to watch.

    python3 service/world.py

Modules 1 to 4 ran on two days of archive. A monitoring module needs more days
than that, and it needs to know which of them changed and how — otherwise there
is no way to check whether a monitor caught the right thing.

So this generates days. Its parameters come from the archive where the archive
has them (`data/calibration.json`, measured in Module 2), and the two kinds of
change it produces are the two the archive actually shows or implies:

  **covariate shift** — the inputs move, the relationship between inputs and
  target is untouched. This is what really happened between 22 and 23 January:
  Module 4 measured mean speed shifting 2.30 reference standard deviations while
  the target moved -0.03. The cause was a person driving 41 per cent of the
  second day against 9 per cent of the first.

  **concept drift** — the inputs are distributed exactly as before and the
  *relationship* changes. Nothing a distribution monitor watches moves at all.
  This one is constructed rather than measured, because the archive is two days
  long and cannot show it. Every number derived from it is tagged `generated`.

The `crew` column is deliberately outside the model's feature set. It explains
the concept drift completely, and it is the point of block four: retraining on
fresh rows is a compromise, while adding the variable that explains the change
repairs the model. In the archive that variable is `mode`, and nobody was
watching it.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CALIBRATION = json.loads((HERE.parent / "data" / "calibration.json").read_text())

SEED = 20200122
ROWS_PER_DAY = 1200
FEATURES = ["speed", "rssi1", "rssi2", "rssiC"]
TARGET = "aboard"

# Regimes, by day number. Days 0-9 are the reference period the model trains on.
STABLE, COVARIATE, CONCEPT = "stable", "covariate shift", "concept drift"


def regime_for(day: int) -> str:
    """Which regime day `day` is in. Fixed, so a check can assert against it."""
    if day < 14:
        return STABLE
    if day < 21:
        # A person starts driving far more often. Speed moves; what makes a
        # passenger 'aboard' does not.
        return COVARIATE
    return CONCEPT


def crew_share(day: int) -> float:
    """Share of readings with a human driving — the archive's `mode`, renamed.

    Measured on the archive: 9.1 per cent on the first day, 41.0 on the second.
    The stable regime sits at the first; the covariate regime moves to the
    second and stays there.
    """
    return 0.091 if regime_for(day) == STABLE else 0.410


def day(number: int, seed: int = SEED) -> pd.DataFrame:
    """One day of rows: four features, the crew flag, and the truth."""
    rng = np.random.default_rng(seed + number)
    regime = regime_for(number)

    aboard_share = CALIBRATION["aboard_share_of_labelled"] / 100
    is_aboard = rng.random(ROWS_PER_DAY) < aboard_share
    with_crew = rng.random(ROWS_PER_DAY) < crew_share(number)

    # Speed: a human drives faster and more variably. The covariate shift arrives
    # entirely through the crew share, which is measured -- 9.1 per cent of the
    # archive's first day against 41.0 of its second. How *much* faster a human
    # drives is a modelling choice, not a measurement, so everything downstream
    # of it is tagged `generated`.
    #
    # Note what happens as that choice grows: the reference period contains crew
    # rows too, so a larger effect inflates the reference's own spread as well as
    # the shifted day's mean, and that inflation is what bounds the standardised
    # shift -- not near the shipped multiplier's 0.6, but well above 0.9 once the
    # effect is pushed to several times its shipped size (measured: 0.60 at this
    # multiplier, 0.77 at 1.5x, 0.85 at 2x, 0.94 at 4x, still climbing). Whatever
    # the true ceiling is, standardising against a contaminated reference bounds
    # it below what an uncontaminated reference would show, which is the property
    # worth knowing before quoting a shift in standard deviations -- the number
    # itself is not 0.6 and depends on how far the effect is pushed.
    base = np.where(is_aboard, rng.normal(2.2, 0.7, ROWS_PER_DAY),
                    rng.normal(0.7, 0.4, ROWS_PER_DAY))
    speed = np.where(with_crew, base * 2.0 + 0.8, base).clip(0, None).round(3)

    # Beacons: heard at the archive's measured rates, and only weakly related to
    # being aboard, because Module 2 measured that the link is not there.
    frame = pd.DataFrame({"day": number, "regime": regime, "crew": with_crew.astype(int),
                          "speed": speed})
    for name in ("rssi1", "rssi2", "rssiC"):
        absent = CALIBRATION["beacon_absent_share"][name] / 100
        strength = rng.normal(-78, 9, ROWS_PER_DAY)
        heard = rng.random(ROWS_PER_DAY) > absent
        frame[name] = np.where(heard, strength.round(1), np.nan)

    if regime == CONCEPT:
        # The relationship inverts, and ONLY where a human is driving. The
        # inputs are distributed exactly as in the covariate regime -- the crew
        # share is the same, the speed distribution is the same -- so every
        # distribution monitor sees nothing at all.
        is_aboard = np.where(with_crew, ~is_aboard, is_aboard)

    frame[TARGET] = is_aboard.astype(int)
    return frame


def blind_day(seed: int = SEED + 999) -> pd.DataFrame:
    """The textbook case: inputs identical to a stable day, relationship inverted.

    The realistic concept drift above arrives alongside a covariate shift, so a
    distribution monitor does fire -- it simply cannot tell the harmful change
    from the harmless one. That is the common case and the more useful lesson.

    This is the pure form, constructed to make the blindness visible on its own:
    every feature is drawn exactly as on a stable day, so every distribution
    measure reads about nought, and the label is flipped for every row. Accuracy
    goes to roughly one minus what it was.

    Not "nothing has moved": the day is a fresh draw, so everything moves a
    little. What is true, and sharper, is that everything a distribution monitor
    watches has moved by *less than the measured quiet floor* -- the largest
    movement this stream produces on a day when nothing at all is happening. A
    threshold above the floor cannot fire on a movement below it, and a threshold
    below the floor fires on every quiet day, so no line can be drawn.
    """
    frame = day(3, seed=seed)          # a stable day, drawn afresh
    frame = frame.copy()
    frame["day"] = -1
    frame["regime"] = "concept drift, inputs unchanged"
    frame[TARGET] = 1 - frame[TARGET]
    return frame


def stream(days: int = 28) -> pd.DataFrame:
    """The whole period, one row per reading."""
    return pd.concat([day(n) for n in range(days)], ignore_index=True)


def reference_period(days: int = 10) -> pd.DataFrame:
    """The days the model is trained on. The monitor's fixed reference."""
    return pd.concat([day(n) for n in range(days)], ignore_index=True)


if __name__ == "__main__":
    everything = stream()
    summary = everything.groupby(["day", "regime"]).agg(
        rows=("speed", "size"),
        mean_speed=("speed", "mean"),
        crew=("crew", "mean"),
        aboard=(TARGET, "mean"),
    ).reset_index()
    print(f"{'day':>4} {'regime':16}{'mean speed':>12}{'crew':>8}{'aboard':>9}")
    for _, row in summary.iterrows():
        if row["day"] % 4 == 0 or row["day"] in (13, 14, 20, 21):
            print(f"{int(row['day']):4} {row['regime']:16}{row['mean_speed']:12.3f}"
                  f"{row['crew']:8.3f}{row['aboard']:9.3f}")
