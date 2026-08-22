"""Lab 1 — What moved, and against what?

Why this lab exists: a model has been serving for ten days and nobody has any
labels, so nobody can compute its accuracy — but the inputs arrive free, every
day, and they can be compared against the days the model was trained on. You
prove here that the comparison only means something when the yardstick is held
still, by building the same monitor twice and watching one of the two go blind.
Where it sits: Block one — "The only free question" and "Which spread? The
question that decides whether the monitor fires", and the definition slides
"Definition — input shift, in reference standard deviations", "Definition — the
fixed reference against the moving baseline" and "Definition — a breach, and the
day it happened on".
What the check grades: input_shift divides by the reference's sample standard
deviation (ddof = 1) and takes the absolute value; both series are twenty-seven
days long and agree with the check's own arithmetic on days 1, 13, 14, 20 and 27;
days_over reports all fourteen days from 14 to 27 on the fixed series and exactly
[14] on the moving one; and which_baseline_goes_blind() returns "moving".
Needs: numpy, and lab_support for the stream and the grain.

Twenty-five minutes.

The model went into service ten days ago. Nobody has looked at it since, because
nothing has broken. Today somebody asks the only question that matters: is the
world still the one it was trained on?

You cannot answer that with the model's accuracy, because you have no labels.
What you do have, free, is the inputs. So compare today's inputs against the
ones the model was trained on and say how far they have moved.

"How far" needs a unit, and the unit is the argument. A difference of 0.75
metres per second means nothing on its own — it is large if the training days
never varied by more than 0.1, and invisible if they swung by 5. So divide by
the spread of the *reference* period:

    shift = | mean(today) - mean(reference) | / standard deviation(reference)

Note which spread. Not today's. The reference is the yardstick, and a yardstick
that changes length every day measures nothing. On this stream, dividing by
today's spread instead reports 0.40 on the days that actually moved 0.60 — below
any threshold you would have set, so the change is silently missed. That is not
a rounding difference, it is a monitor that does not fire.

Then the second half, which is the point of the block. There are two ways to
choose a reference:

    fixed       always compare against the ten training days
    moving      compare against yesterday

The moving baseline is tempting: it needs no stored reference, it adapts, it is
what most quick implementations do. Measure both over the twenty-seven served
days and count how often each says something. Then answer, in
which_baseline_goes_blind(), which one stops reporting a world that is still
wrong — and be ready to say why in one sentence.

What you write: input_shift, shift_against_fixed, shift_against_yesterday,
days_over, and the one-word answer.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from lab_support import (NotSolved, DAYS, REFERENCE_DAYS, WATCHED,   # noqa: E402
                         SHIFT_THRESHOLD, reference_frame, day_frame)

LAB = 1


def input_shift(reference_values, current_values) -> float:
    """How far `current_values` has moved, in reference standard deviations.

    Both arguments are one-dimensional arrays of the same measurement. Return a
    single non-negative number. Use the sample standard deviation (ddof=1) of
    the reference, and take the absolute value — a monitor cares that the world
    moved, not which way.

    Definition graded by the check:
        Δ_t = |mean_t − mean_ref| / s_ref, with s_ref the sample standard deviation (ddof = 1) of the reference period
        (Glass, 1976; Gama et al., 2014, §2.1–2.2). Choices: the reference's
        spread and never the current day's; ddof = 1, the sample estimate; the
        absolute value, so a fall does not cancel a rise. Slide: "Definition —
        input shift, in reference standard deviations".
    Needs: numpy
    """
    # TODO: | mean(current) - mean(reference) | / sd(reference)
    raise NotSolved("input_shift(reference_values, current_values) still raises "
                    "instead of returning a number")


def shift_against_fixed(days: int = DAYS) -> list:
    """The shift of each served day against the fixed reference period.

    Served days are REFERENCE_DAYS..days-1 is *not* what is wanted here: the
    monitor runs from day 1 onward, including days inside the reference period,
    because on those days it should say 'nothing has moved' and you want to see
    that it does. Return a list of length days-1, for days 1 to days-1, each
    entry the shift of that day's WATCHED column against the reference frame's.

    Definition graded by the check:
        fixed: Δ_t = |mean_t − mean_ref| / s_ref, one reference period held still
        (Page, 1954; Gama et al., 2014, §3.2). Choices: the reference period is
        days 0 to REFERENCE_DAYS − 1, read once and never updated; the watched
        column is WATCHED. Slide: "Definition — the fixed reference against the
        moving baseline".
    Needs: reference_frame, day_frame, input_shift, pandas
    """
    # TODO: reference_frame() once, then day_frame(n) for each n.
    raise NotSolved("shift_against_fixed(days) still raises instead of returning a list")


def shift_against_yesterday(days: int = DAYS) -> list:
    """The same measure, with yesterday standing in for the reference.

    Same length, same order. For day n, the reference is day n-1 — so the
    yardstick is yesterday's spread, and it moves every day.

    Definition graded by the check:
        moving: Δ_t = |mean_t − mean_{t−1}| / s_{t−1}, the yardstick redrawn every day
        (Page, 1954; Gama et al., 2014, §3.2). Choice: yesterday supplies both
        the centre and the spread. Slide: "Definition — the fixed reference
        against the moving baseline".
    Needs: day_frame, input_shift, pandas
    """
    # TODO: day_frame(n - 1) as the reference for day n.
    raise NotSolved("shift_against_yesterday(days) still raises instead of "
                    "returning a list")


def days_over(series, threshold: float = SHIFT_THRESHOLD) -> list:
    """Which days reached the threshold. Return day numbers, not list positions.

    The series starts at day 1, so position 0 is day 1. Getting this off by one
    is the most common mistake in the lab and the check will tell you plainly.

    Definition graded by the check:
        over(Δ, τ) = { t : Δ_t ≥ τ }, reported as day numbers, with day 1 at position 0
        (Gama et al., 2014, §3.2). Choice: at or over the line counts, not
        strictly over — which is why Lab 3's safe threshold must clear the quiet
        floor rather than match it. Slide: "Definition — a breach, and the day it
        happened on".
    Needs: enumerate
    """
    # TODO: enumerate, compare, convert position to day number.
    raise NotSolved("days_over(series, threshold) still raises instead of "
                    "returning day numbers")


def which_baseline_goes_blind() -> str:
    """Return "fixed" or "moving" — whichever stops reporting a changed world.

    Run this file. One of the two lists is long and one has a single entry.
    Decide which behaviour is the failure. The world does not go back to normal
    on day 15; only one of these two monitors thinks it did.

    Definition graded by the check:
        blind: Δ_t returns to the quiet floor after a step, though mean_t ≠ mean_ref throughout
        (Page, 1954; Gama et al., 2014, §3.2). Choice: "blind" means reporting
        calm while the world is still changed, not reporting less often. Slide:
        "Definition — the fixed reference against the moving baseline".
    Needs: nothing — one word, from what you measured
    """
    # TODO: one word, from what you measured.
    raise NotSolved("which_baseline_goes_blind() still raises instead of returning "
                    '"fixed" or "moving"')


if __name__ == "__main__":
    fixed = shift_against_fixed()
    moving = shift_against_yesterday()
    print(f"watching '{WATCHED}', reference = days 0 to {REFERENCE_DAYS - 1}, "
          f"threshold = {SHIFT_THRESHOLD} reference SD\n")
    print(f"{'day':>4}{'against fixed':>16}{'against yesterday':>20}")
    for number, (a, b) in enumerate(zip(fixed, moving), start=1):
        if number in (1, 5, 10, 13, 14, 15, 16, 20, 21, 27):
            print(f"{number:4}{a:16.3f}{b:20.3f}")
    print(f"\nover the line, fixed    : {days_over(fixed)}")
    print(f"over the line, moving   : {days_over(moving)}")
    print(f"goes blind              : {which_baseline_goes_blind()}")
