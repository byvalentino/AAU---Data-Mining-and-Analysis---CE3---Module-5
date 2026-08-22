"""Lab 3 — One change should page one person, once.

Why this lab exists: Lab 1's monitor is correct and fires on fourteen days about
one event, which is how a monitoring channel gets muted by somebody who was
right to mute it. You prove here that a threshold can be justified against a
measured noise floor rather than chosen by taste, and that confirmation and a
cooldown turn fourteen true pages into three at a cost of exactly one day, which
you measure rather than assume.
Where it sits: Block three — "Decision one — where the line goes, and where the
number comes from", "The same signal, and the number of times somebody is
woken", "Forty columns, and two alarms a day from nothing" and "What an alert
should say", and the definition slides "Definition — the quiet floor and the
lowest safe threshold", "Definition — confirmation, the cooldown, and what they
cost", "Definition — watching many columns: the family floor and the expected
false alarms" and "Definition — the alert verdict: page, ticket or nothing".
What the check grades: quiet_floor is the maximum over days 1 to last_quiet_day
inclusive; pages_with_confirmation turns the fourteen raw breaches on the real
series into pages on days 15, 20 and 25 and resets its run on any day under the
line; detection_delay reports the days lost to confirmation — including on a
series whose run breaks, where the answer is not simply confirmations minus one
— and -1 for a series that never crosses; lowest_safe_threshold returns 0.10
on the real series and None when every candidate fires on a quiet day;
shift_by_column reproduces Lab 1's series for speed and gives one series per
watched column, family_quiet_floor over those four is 0.1507 against speed's
0.0573 and expected_false_alarms is one multiplication; and alert_verdict
returns the call the evidence justifies on six situations whose right calls
differ, with a reason built out of the numbers it was handed.
Needs: lab_support for Lab 1's series and the shipped alerting settings.

Twenty-five minutes.

Lab 1 gave you a number for every day and a line to compare it against. Fourteen
of the twenty-seven days are over the line, so fourteen pages go out. Every one
of them is true. Every one of them is about the same change, which somebody
already knows about, and after the third the person on call stops reading them.

An alert that fires every day is not a monitor, it is weather. The work in this
block is turning a correct signal into a useful one, and it is three decisions:

    the threshold      where the line goes, and where that number comes from
    confirmation       how many days in a row before anybody is woken
    the cooldown       how long the monitor stays quiet after it has spoken

Start with the threshold, because it is the one people invent. The shift is not
nought on a quiet day — sampling alone moves it. So measure how large it gets
while nothing at all is happening, and put the line above that. A threshold
chosen after seeing the answer is a decoration; a threshold set above a measured
quiet floor is a decision you can defend in a review.

Then confirmation, which costs you something. Requiring two days in a row means
you hear about a real change one day later than you could have. Measure that
delay rather than assuming it, and decide whether a day is worth the silence.

Then the part that is not arithmetic. A correct signal, confirmed and cooled
down, still has to become a decision: wake somebody, write a ticket, or say
nothing. That call is the last function in this file, and its *reason* is graded
as hard as the call is — because a call with no argument behind it is a coin
flip, and this course is not examining coin flips.

Then the part almost every monitor gets wrong within a month of going live: it
watches more than one column. Block one built the monitor on one, which was
enough to show what a yardstick is. Point the same monitor at all four inputs
and measure the floor again — the floor a threshold has to clear is now the
worst quiet day of the worst column, and it is nearly three times higher. Then
do the multiplication for forty columns, which is what you will actually be
handed, and see what a per-column level of five per cent costs per day.

What you write: quiet_floor, pages_with_confirmation, detection_delay,
lowest_safe_threshold, shift_by_column, family_quiet_floor,
expected_false_alarms and alert_verdict.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lab_support import (NotSolved, CONFIRMATIONS, COOLDOWN, DAYS,    # noqa: E402
                         SHIFT_THRESHOLD, WATCHED_COLUMNS, day_frame,
                         load_lab, reference_frame)

LAB = 3

# The last day before anything changes. Days 1 to 13 are the quiet stretch, and
# the shift measured on them is the floor a threshold must clear.
LAST_QUIET_DAY = 13

# The thresholds on offer, from one that fires on almost anything to one that
# almost never does. A round ladder rather than a number fitted to the data,
# because a threshold has to survive being read aloud in a review: "we alert at
# 0.10 reference standard deviations, and the measured quiet floor is 0.06."
CANDIDATES = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]

# The three calls an alert verdict may make, and there is no fourth. `page`
# wakes somebody now; `ticket` is written down and read in working hours;
# `nothing` is the answer a monitor has to be allowed to give, or it becomes a
# monitor that only ever finds drift and is therefore ignored.
CALLS = ("page", "ticket", "nothing")


def quiet_floor(series, last_quiet_day: int = LAST_QUIET_DAY) -> float:
    """The largest shift seen while nothing was changing.

    `series` is Lab 1's fixed-reference shift, one entry per day starting at
    day 1. Return the maximum over days 1 to last_quiet_day inclusive.

    This number is the whole reason a threshold can be justified. Quote it
    beside any threshold you set.

    Definition graded by the check:
        floor = max_{1 ≤ t ≤ q} Δ_t over the quiet days, day 1 at position 0
        (Beyer et al., 2016, ch. 6; Page, 1954). Choices: the maximum and not
        the mean, because the monitor fires on the worst quiet day rather than
        on the average one; the quiet stretch is days 1 to LAST_QUIET_DAY, fixed
        before any threshold was chosen. Slide: "Definition — the quiet floor and
        the lowest safe threshold".
    Needs: max, slicing
    """
    # TODO: the maximum over the quiet stretch. Mind the day-to-position offset.
    raise NotSolved("quiet_floor(series, last_quiet_day) still raises instead of "
                    "returning a number")


def pages_with_confirmation(series, threshold: float = SHIFT_THRESHOLD,
                            confirmations: int = CONFIRMATIONS,
                            cooldown: int = COOLDOWN) -> list:
    """Which days somebody is actually woken. Return day numbers.

    Walk the series day by day, keeping a count of consecutive days at or over
    the threshold. A day pages when both of these hold:

        the run of consecutive days over the line has reached `confirmations`
        at least `cooldown` days have passed since the last page

    A day under the threshold resets the run to nought. The first page is never
    blocked by a cooldown, because there is nothing before it.

    With the shipped settings — threshold 0.5, two confirmations, five days of
    cooldown — this turns fourteen pages into three.

    Definition graded by the check:
        run_t = run_{t−1} + 1 if Δ_t ≥ τ else 0; page at t ⟺ run_t ≥ k and t ≥ p + c, p the last paging day
        (Page, 1954; Beyer et al., 2016, ch. 6). Choices: consecutive and never
        cumulative, so one day under the line resets the run; the first page has
        no previous page, so no cooldown can block it. Slide: "Definition —
        confirmation, the cooldown, and what they cost".
    Needs: enumerate
    """
    # TODO: one pass, two pieces of state: the run, and the day the quiet ends.
    raise NotSolved("pages_with_confirmation(series, threshold, confirmations, "
                    "cooldown) still raises instead of returning day numbers")


def detection_delay(series, threshold: float = SHIFT_THRESHOLD,
                    confirmations: int = CONFIRMATIONS) -> int:
    """How many days later you hear about it, because of confirmation.

    Return the first paging day minus the first day the series crossed the
    threshold. Return 0 if the two are the same day, and -1 if the series never
    crosses at all, so a caller can tell 'immediate' from 'never'.

    The cooldown does not appear here, and cannot: there is no previous page for
    it to measure from, so it never delays the first one. Only confirmation
    costs you time at the start.

    Definition graded by the check:
        delay = first paging day − first breach, and −1 when the series never crosses
        (Page, 1954; Gama et al., 2014, §3.2). Choice: measured on this series
        rather than quoted as confirmations − 1, which is only the same number
        while the run never breaks. Slide: "Definition — confirmation, the
        cooldown, and what they cost".
    Needs: pages_with_confirmation, enumerate
    """
    # TODO: the first crossing, the first page, the difference.
    raise NotSolved("detection_delay(series, threshold, confirmations) still raises "
                    "instead of returning a number of days")


def lowest_safe_threshold(series, last_quiet_day: int = LAST_QUIET_DAY,
                          candidates: list = None) -> float:
    """The smallest candidate threshold that says nothing during the quiet days.

    Sort `candidates` into increasing order and return the first one for which
    no day in 1..last_quiet_day reaches it. `candidates` of None means the
    CANDIDATES ladder at the top of this file, so that the function can be
    called with the series alone — which is how the check calls it. Return None
    if every candidate fires on a quiet day, which is the honest answer when the
    floor is as large as the signal, and it happens.

    Lower is better, because a lower line notices a smaller change. The floor is
    what stops you going lower, and it is measured, not chosen.

    Definition graded by the check:
        τ* = min { τ ∈ C : τ > floor }, and None when no candidate clears the floor
        (Beyer et al., 2016, ch. 6). Choices: strictly above the floor, because a
        breach is "at or over" the line, so a threshold equal to the floor fires
        on the quiet day that produced it; None rather than the largest
        candidate, because "there is no safe line" is a finding. Slide:
        "Definition — the quiet floor and the lowest safe threshold".
    Needs: sorted, quiet_floor

    A note on binning, since most drift indices need it and this one does not.
    Module 4 measured that at twenty bins most of its divergence index is the
    small constant put under an empty bin's share to keep the logarithm finite,
    rather than the data. The shift this module watches is a difference of means
    divided by a spread, so it bins nothing at all; where this module does bin —
    the five bands behind the blind case on the appendix slide — it stays at or
    below ten bins and prints the count beside the number.
    """
    # TODO: the first candidate above the quiet floor.
    raise NotSolved("lowest_safe_threshold(series, last_quiet_day, candidates) still "
                    "raises instead of returning a threshold or None")


def shift_by_column(days: int = DAYS, columns: list = None) -> dict:
    """Lab 1's monitor, pointed at every watched column instead of one.

    `columns` of None means WATCHED_COLUMNS, the four inputs the model is
    actually fed. Return a dict from column name to a list of length days-1 —
    the same shape Lab 1's shift_against_fixed returns, one series per column,
    each measured against the same fixed reference period.

    Use Lab 1's input_shift, through load_lab(1), so that one measure is
    defined once. Drop the rows where a column is absent before taking either
    mean: an absent beacon reading is not a low reading, and *where* a column is
    absent is its own monitor, which Module 2 built.

    The series for "speed" must come out identical to Lab 1's
    shift_against_fixed() — same reference, same measure, one column of many.

    Definition graded by the check:
        Δ_{c,t} = |mean_{c,t} − mean_{c,ref}| / s_{c,ref} for each watched column c, missing readings dropped
        (Rabanser, Günnemann & Lipton, 2019). Choices: the same fixed reference
        period for every column; missing readings dropped rather than filled, so
        an absence does not read as a movement. Slide: "Definition — watching
        many columns: the family floor and the expected false alarms".
    Needs: reference_frame, day_frame, WATCHED_COLUMNS, load_lab, pandas
    """
    # TODO: one reference per column, then the same measure day by day.
    raise NotSolved("shift_by_column(days, columns) still raises instead of returning "
                    "one series per column")


def family_quiet_floor(series_by_column: dict,
                       last_quiet_day: int = LAST_QUIET_DAY) -> float:
    """The quiet floor of the whole family, not of one column.

    `series_by_column` is what shift_by_column returns. Return the largest
    movement *any* watched column reaches on *any* of the quiet days — so the
    floor a threshold has to clear when the monitor is allowed to fire on any of
    them.

    This is the number that makes multiplicity concrete. Watch one column and
    you clear one column's noise; watch four and you have to clear the worst of
    four; watch forty and the floor is the worst of forty. Nothing about the
    measure changed — the number of chances to be unlucky did.

    Definition graded by the check:
        floor_family = max_c max_{1 ≤ t ≤ q} Δ_{c,t}, the largest quiet-day movement any watched column makes
        (Rabanser, Günnemann & Lipton, 2019; Beyer et al., 2016, ch. 6). Choice:
        the maximum over columns as well as over days, because the alert fires
        when any column fires. Slide: "Definition — watching many columns: the
        family floor and the expected false alarms".
    Needs: quiet_floor, max
    """
    # TODO: the worst quiet day of the worst column.
    raise NotSolved("family_quiet_floor(series_by_column, last_quiet_day) still raises "
                    "instead of returning a number")


def expected_false_alarms(columns: int, level: float, days: int = 1) -> float:
    """How many alarms a day of nothing at all is expected to produce.

    `columns` watched, each with a per-column threshold that fires on a share
    `level` of days when nothing is happening, over `days` days. Return the
    expected count.

    This is the arithmetic a graduate meets in their first week and nobody warns
    them about. It is one multiplication, and it is the reason a monitoring
    channel that watched one column happily and forty columns unusably is not a
    bug in anybody's code.

    Definition graded by the check:
        E[alarms] = m · α · d for m columns at a per-column level α over d days
        (Rabanser, Günnemann & Lipton, 2019). Choices: expected count rather than
        the probability of at least one, because a person on call counts
        messages; the columns are treated as independent, which flatters the
        count when they move together. Slide: "Definition — watching many
        columns: the family floor and the expected false alarms".
    Needs: nothing — one multiplication
    """
    # TODO: one multiplication, and then look at what it says about forty columns.
    raise NotSolved("expected_false_alarms(columns, level, days) still raises instead "
                    "of returning a number of alarms")


def alert_verdict(evidence: dict) -> tuple:
    """Return (call, reason). `call` is one of "page", "ticket", "nothing".

    This is the slide "What an alert should say", written down as a function so
    that it can be argued with. `evidence` is what you measured, and it holds
    exactly these five keys:

        quiet_floor           the largest movement this monitor reports while
                              nothing at all is happening — your own Lab 3 number
        observed              the movement actually seen, on the same scale
        detection_delay_days  what confirmation cost, in days, and -1 when the
                              series never crossed at all
        truth_has_arrived     True when bought labels for this period are in
                              hand, False when nobody has checked a row yet
        level                 "input", "output" or "outcome" — which of block
                              two's three levels the movement was seen at

    Three questions, in this order, and the order is the content:

        1. Is the movement inside the quiet floor? Then it is what this monitor
           does on a day when nothing is happening. Call `nothing`.
        2. Is it at the outcome level, with truth in hand? Then somebody is
           being served worse, it is established, and it will not improve by
           itself. Call `page`.
        3. Otherwise call `ticket`. An input or output movement is a reason to
           spend a morning buying truth, not a reason to wake anybody: nobody
           can act at three in the morning on a number that may be noise.

    The reason is graded as hard as the call. It must be an argument somebody
    who was not in the room could disagree with, so: at least forty characters,
    every number in it one of your own measurements from `evidence`, and at
    least two of the quantities you weighed named. A sentence copied off a slide
    fails, because the slide's numbers are not this evidence's numbers.

    Do not modify `evidence`. A verdict reads its evidence; it does not edit it.

    Definition graded by the check:
        verdict(evidence) = nothing if observed ≤ quiet_floor · page if level = outcome and truth has arrived · ticket otherwise, with a reason naming ≥ 2 of the quantities weighed
        (Beyer et al., 2016, ch. 6; Breck et al., 2017). Choices: "at or under
        the floor" counts as nothing, because the floor is the largest movement
        a quiet day produced and no safe threshold can sit at it; only the
        outcome level pages, because it is the only level that says somebody was
        served worse; and the reason is graded, because a call with no argument
        is a coin flip. Slide: "Definition — the alert verdict: page, ticket or
        nothing".
    Needs: the five keys above, and nothing else
    """
    # TODO: three questions in order, then a reason built from `evidence`.
    raise NotSolved("alert_verdict(evidence) still raises instead of returning "
                    "(call, reason)")


if __name__ == "__main__":
    lab1 = load_lab(1)
    series = lab1.shift_against_fixed()

    floor = quiet_floor(series)
    raw = lab1.days_over(series, SHIFT_THRESHOLD)
    paged = pages_with_confirmation(series)

    print(f"quiet floor, days 1 to {LAST_QUIET_DAY}   : {floor:.4f}")
    print(f"threshold in use                : {SHIFT_THRESHOLD} "
          f"({SHIFT_THRESHOLD / floor:.0f} times the floor)")
    print(f"lowest safe candidate           : {lowest_safe_threshold(series)}")
    print()
    print(f"threshold alone                 : {len(raw)} pages  {raw}")
    print(f"with {CONFIRMATIONS} confirmations, {COOLDOWN}-day cooldown : "
          f"{len(paged)} pages  {paged}")
    print(f"cost of confirmation            : {detection_delay(series)} day(s) later")

    # The verdict on the day the monitor first spoke, from your own numbers.
    print()
    for label, evidence in (
            ("day 15, the inputs stepped, nobody has bought a label",
             {"quiet_floor": floor, "observed": series[14],
              "detection_delay_days": detection_delay(series),
              "truth_has_arrived": False, "level": "input"}),
            ("day 15, and nothing has moved further than a quiet day does",
             {"quiet_floor": floor, "observed": series[4],
              "detection_delay_days": detection_delay(series),
              "truth_has_arrived": False, "level": "input"})):
        call, reason = alert_verdict(evidence)
        print(f"{label}\n  -> {call}: {reason}")
