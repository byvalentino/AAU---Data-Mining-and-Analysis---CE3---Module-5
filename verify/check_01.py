#!/usr/bin/env python3
"""Check 1 — the yardstick, and the baseline that goes blind."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, explain, not_ready                   # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    import numpy as np                                                # noqa: E402
    from lab_support import (DAYS, WATCHED, SHIFT_THRESHOLD,          # noqa: E402
                             reference_frame, day_frame)
except ImportError as unready:
    not_ready(unready)


def body(lab):
    # ---- the measure itself, on numbers the check controls -------------------
    reference = np.array([0.0, 1.0, 2.0, 3.0, 4.0])        # mean 2, sd(ddof=1) = sqrt(2.5)
    close(lab.input_shift(reference, reference + 1.0), 1 / np.sqrt(2.5), 1e-9,
          "input_shift on a reference of 0..4 moved up by exactly one")

    # Down must read the same as up. A signed measure lets a fall cancel a rise.
    close(lab.input_shift(reference, reference - 1.0), 1 / np.sqrt(2.5), 1e-9,
          "input_shift when the current day moved DOWN by one — a monitor cares "
          "that the world moved, not which way, so take the absolute value")

    # Divide by the reference's spread, not the current day's. Here the current
    # day is four times as wide, so the two choices differ by a factor of four.
    wide = np.array([-6.0, -3.0, 0.0, 3.0, 6.0]) + 3.0     # mean 3, sd four times larger
    close(lab.input_shift(reference, wide), 1 / np.sqrt(2.5), 1e-9,
          "input_shift when the current day is four times as wide as the reference. "
          "The reference is the yardstick; dividing by the current day's spread "
          "would report a quarter of this")

    # Population versus sample standard deviation, stated so nobody guesses.
    close(lab.input_shift(np.array([0.0, 2.0]), np.array([1.0, 3.0])), 1 / np.sqrt(2), 1e-9,
          "input_shift on a two-value reference — use ddof=1, the sample standard "
          "deviation, as the docstring says")

    # ---- the two series ------------------------------------------------------
    fixed = list(lab.shift_against_fixed())
    moving = list(lab.shift_against_yesterday())
    assert len(fixed) == DAYS - 1, (
        f"shift_against_fixed returned {len(fixed)} values; expected {DAYS - 1}, "
        "one for each of days 1 to 27")
    assert len(moving) == DAYS - 1, (
        f"shift_against_yesterday returned {len(moving)} values; expected {DAYS - 1}")

    reference_values = reference_frame()[WATCHED].to_numpy()
    spread = reference_values.std(ddof=1)
    for number in (1, 13, 14, 20, 27):
        today = day_frame(number)[WATCHED].to_numpy()
        expected = abs(today.mean() - reference_values.mean()) / spread
        close(fixed[number - 1], expected, 1e-9, f"the fixed-reference shift on day {number}")
        yesterday = day_frame(number - 1)[WATCHED].to_numpy()
        expected = abs(today.mean() - yesterday.mean()) / yesterday.std(ddof=1)
        close(moving[number - 1], expected, 1e-9, f"the moving-baseline shift on day {number}")

    # ---- the alerting days, and the off-by-one -------------------------------
    over_fixed = list(lab.days_over(fixed))
    over_moving = list(lab.days_over(moving))
    assert over_fixed and min(over_fixed) == 14, (
        f"days_over on the fixed series gave {over_fixed[:4]}...; the first day over "
        "the line is day 14, the day the crew share steps up. Getting 13 means you "
        "returned the list position instead of the day number")
    assert len(over_fixed) == 14, (
        f"days_over on the fixed series found {len(over_fixed)} days; expected 14 — "
        "every day from 14 to 27. Finding none usually means you divided by the "
        "current day's spread, which reports about 0.40 where the true shift is 0.60")
    assert over_moving == [14], (
        f"days_over on the moving series gave {over_moving}; expected exactly [14]. "
        "The moving baseline sees the step on the day it happens and nothing after")

    # ---- the one-word answer -------------------------------------------------
    answer = lab.which_baseline_goes_blind().strip().lower()
    assert answer in ("fixed", "moving"), (
        f'which_baseline_goes_blind() returned "{answer}"; expected "fixed" or "moving"')
    # The failure message used to carry the whole argument, so a student could
    # type "fixed", read why that was wrong, type "moving" and pass without ever
    # having decided anything. The reasoning is now withheld until the third
    # failure of this same assertion: the first two say only that the answer is
    # wrong, which is what a question sounds like.
    assert answer == "moving", explain(
        "m5:blind",
        'which_baseline_goes_blind() returned "fixed", and that is not the answer',
        f"The fixed reference keeps reporting the shift on all {len(over_fixed)} "
        "changed days, which is a nuisance. The moving baseline reports it once, on "
        "day 14, and then compares each shifted day against another shifted day -- "
        "so from day 15 it calls the new world normal while the model goes on being "
        "wrong. Going blind is reporting calm while the world is still changed, not "
        "reporting less often.")

    # The lesson is the gap between the two counts, so insist it is a real gap.
    assert len(over_fixed) >= 10 * len(over_moving), (
        f"the fixed reference should keep firing far longer than the moving one "
        f"({len(over_fixed)} days against {len(over_moving)}); yours were "
        f"{len(over_fixed)} and {len(over_moving)}")


run(1, "01_what_moved", "input_shift", body)
