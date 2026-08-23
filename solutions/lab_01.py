"""Lab 1 — solution. What moved, and against what?"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from lab_support import (NotSolved, DAYS, REFERENCE_DAYS, WATCHED,   # noqa: F401
                         SHIFT_THRESHOLD, reference_frame, day_frame)

LAB = 1


def input_shift(reference_values, current_values) -> float:
    # The reference's spread is the yardstick. Using the current day's instead
    # lets a day that widens its own spread hide the fact that it also moved.
    reference = np.asarray(reference_values, dtype=float)
    current = np.asarray(current_values, dtype=float)
    return float(abs(current.mean() - reference.mean()) / reference.std(ddof=1))


def shift_against_fixed(days: int = DAYS) -> list:
    reference = reference_frame()[WATCHED].to_numpy()
    return [input_shift(reference, day_frame(number)[WATCHED].to_numpy())
            for number in range(1, days)]


def shift_against_yesterday(days: int = DAYS) -> list:
    return [input_shift(day_frame(number - 1)[WATCHED].to_numpy(),
                        day_frame(number)[WATCHED].to_numpy())
            for number in range(1, days)]


def days_over(series, threshold: float = SHIFT_THRESHOLD) -> list:
    # The series starts at day 1, so position 0 is day 1.
    return [position + 1 for position, value in enumerate(series) if value >= threshold]


def which_baseline_goes_blind() -> str:
    # The moving baseline compares each day against one just like it. After the
    # step, every day matches the day before, so the difference falls back to
    # nought and the monitor reports calm while the world stays wrong.
    return "moving"


if __name__ == "__main__":
    import pandas as pd
    import plotly.graph_objects as go

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from _narrate import narrator, show_table, save_figure

    say = narrator(LAB)
    say.info("Lab 1 — the same monitor twice: one yardstick held still, one redrawn "
             "every day, and only one of them still sees the change on day 22")

    reference = reference_frame()
    say.info("reference period: days 0 to %d, %s rows, generated (seed 20200122) and "
             "read from data/stream.parquet", REFERENCE_DAYS - 1, f"{len(reference):,}")
    values = reference[WATCHED].to_numpy()
    say.info("what 'normal' means for '%s': mean %.4f m/s, sample standard deviation "
             "(ddof=1) %.4f m/s — the denominator of every number below",
             WATCHED, values.mean(), values.std(ddof=1))

    fixed = shift_against_fixed()
    moving = shift_against_yesterday()
    say.info("measured %d served days, 1 to %d, both ways", len(fixed), DAYS - 1)

    over_fixed, over_moving = days_over(fixed), days_over(moving)
    say.info("against the fixed reference: %d day(s) at or over %.2f reference standard "
             "deviations — %s", len(over_fixed), SHIFT_THRESHOLD, over_fixed)
    say.info("against yesterday: %d day(s) — %s. The world did not go back to normal on "
             "day 15; only this monitor thinks it did",
             len(over_moving), over_moving)
    say.info("the step itself, day 14: fixed %.3f, moving %.3f. Both see it. The "
             "difference is everything that comes after", fixed[13], moving[13])
    say.info("day 22, deep inside the changed world: fixed %.3f, moving %.3f",
             fixed[21], moving[21])

    table = pd.DataFrame({
        "day": range(1, DAYS),
        "regime": [day_frame(n)["regime"].iloc[0] for n in range(1, DAYS)],
        "against_fixed_sd": np.round(fixed, 4),
        "against_yesterday_sd": np.round(moving, 4),
    })
    show_table(table, "the shift of every served day, both yardsticks "
                      "(reference standard deviations)", logger=say)

    days = list(range(1, DAYS))
    figure = go.Figure()
    figure.add_hline(y=SHIFT_THRESHOLD, line=dict(color="#52514E", dash="dash", width=1.5),
                     annotation_text=f"alerting threshold, {SHIFT_THRESHOLD} reference s.d.",
                     annotation_position="top left")
    figure.add_trace(go.Scatter(x=days, y=fixed, mode="lines+markers",
                                name="against a fixed reference",
                                line=dict(color="#2A78D6", width=2)))
    figure.add_trace(go.Scatter(x=days, y=moving, mode="lines+markers",
                                name="against yesterday",
                                line=dict(color="#E07B39", width=2)))
    figure.update_layout(
        title="A moving baseline notices once, then calls the new world normal",
        xaxis_title="day of the served month (day)",
        yaxis_title="input shift (reference standard deviations)",
        legend=dict(x=0.02, y=0.98))
    save_figure(figure, "fixed_against_moving", LAB, logger=say)

    say.info("the answer the check grades: which_baseline_goes_blind() = %r",
             which_baseline_goes_blind())
    say.info("what the check grades: input_shift divides by the reference's ddof=1 "
             "spread and is absolute; the two series are %d days long and match the "
             "check's own arithmetic on days 1, 13, 14, 20 and 27; days_over gives all "
             "fourteen days from 14 to 27 on the fixed series and exactly [14] on the "
             "moving one", DAYS - 1)
