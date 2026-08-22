#!/usr/bin/env python3
"""Check 3 — the threshold, the confirmation, the cooldown, and the call."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, explain, grade_reason                # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (SHIFT_THRESHOLD, WATCHED_COLUMNS,            # noqa: E402
                         day_frame, load_lab, reference_frame)

CALLS = ("page", "ticket", "nothing")

# Six situations whose right calls differ, and the numbers in them are this
# module's own: the quiet floor Lab 3 measures, the stable week's input level,
# the covariate week's, the output-rate step between those two weeks, and the
# accuracy the concept week lost. A student who has run the labs recognises
# every one of them.
#
# The call is a choice from three, so on its own it is a coin flip. What makes
# this an examination is that the arithmetic is nearly identical across the last
# four and the call is not: the decision rests on which level moved and on
# whether anybody has bought truth yet, and no lookup keyed on the numbers
# survives the perturbations below.
VERDICTS = (
    ("a movement inside the quiet floor",
     {"quiet_floor": 0.0573, "observed": 0.0084, "detection_delay_days": -1,
      "truth_has_arrived": False, "level": "input"},
     "nothing",
     "The floor is the largest movement this monitor reports on a day when "
     "nothing at all is happening. A movement smaller than that is not evidence "
     "of anything, and no threshold that clears the floor could have fired on "
     "it. 'Nothing' has to be a call you are willing to make, or the monitor "
     "only ever finds drift and is switched off."),

    ("the model's own output moved, and nobody has bought a label yet",
     {"quiet_floor": 0.0573, "observed": 0.136, "detection_delay_days": 1,
      "truth_has_arrived": False, "level": "output"},
     "ticket",
     "Something real moved: the movement is well clear of the floor. But no "
     "label has been checked, so nobody yet knows whether it cost anything. "
     "Nobody can act at three in the morning on a number that may be noise, and "
     "what this justifies is a morning of buying truth."),

    ("bought truth says the outcome level has fallen",
     {"quiet_floor": 0.0573, "observed": 0.105, "detection_delay_days": 1,
      "truth_has_arrived": True, "level": "outcome"},
     "page",
     "This is the only one of the six in which somebody is measurably being "
     "served worse and the measurement is in hand. The outcome level is the "
     "only level that says that, and bought truth is what establishes it. It "
     "will not repair itself overnight."),

    ("a large input movement whose output and outcome are unchanged",
     {"quiet_floor": 0.0573, "observed": 0.615, "detection_delay_days": 1,
      "truth_has_arrived": False, "level": "input"},
     "ticket",
     "This is the commonest case in production and the one that trains people "
     "to ignore the channel. The world moved; nothing yet says the model is "
     "worse for it. Module 4's verdict on the real archive was exactly this "
     "shape, and the right action there was none."),

    ("the outcome level looks lower, but no truth has been bought",
     {"quiet_floor": 0.0573, "observed": 0.105, "detection_delay_days": 1,
      "truth_has_arrived": False, "level": "outcome"},
     "ticket",
     "The same movement, at the same level, as the situation that pages, and "
     "one thing differs: nobody has checked a row. Without bought truth there "
     "is no measured loss, only a number. Buy the labels first."),

    ("an outcome movement exactly the size of the quiet floor",
     {"quiet_floor": 0.0573, "observed": 0.0573, "detection_delay_days": 0,
      "truth_has_arrived": True, "level": "outcome"},
     "nothing",
     "A movement equal to the floor is the size of the largest quiet day. Lab "
     "3's own rule says a candidate threshold has to clear the floor rather "
     "than match it, because a breach is 'at or over' the line, so a threshold "
     "that could fire on this would fire on the quiet day that produced the "
     "floor."),
)

# One quantity changed, and the call has to change with it. This is what a
# lookup table cannot survive: the same situation, one number or one flag
# different, a different answer.
PERTURBED = (
    ("bought truth says the outcome level has fallen", "truth_has_arrived", False,
     "ticket",
     "Nothing about the movement changed. What changed is whether anybody has "
     "checked a row, and that is what turns a number into a measured loss."),
    ("bought truth says the outcome level has fallen", "quiet_floor", 0.5,
     "nothing",
     "The same measured loss, watched by a far noisier monitor. When the floor "
     "is above the movement, the movement is inside what this monitor does "
     "while nothing is happening."),
    ("a large input movement whose output and outcome are unchanged", "level",
     "outcome", "ticket",
     "The level moved to the one that pages and still nobody has bought a "
     "label. Both conditions have to hold before anybody is woken."),
    ("a large input movement whose output and outcome are unchanged", "observed",
     0.0084, "nothing",
     "The same level and the same absent truth, and now a movement smaller than "
     "the floor."),
)


def one_verdict(lab, label, evidence, expected, because, key):
    """Grade one call and one reason, and refuse a verdict that edits its evidence."""
    handed = dict(evidence)
    result = lab.alert_verdict(handed)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"alert_verdict() returned {result!r} on {label}; it returns the pair "
        "(call, reason)")
    call, reason = result
    assert call in CALLS, (
        f"alert_verdict() called {call!r} on {label}. The three calls are "
        f"{', '.join(repr(c) for c in CALLS)}, spelled exactly like that")
    assert handed == evidence, (
        f"alert_verdict() changed the evidence dictionary it was handed while judging "
        f"{label}. A verdict reads its evidence; it does not edit it")
    again, _ = lab.alert_verdict(dict(evidence))
    assert again == call, (
        f"alert_verdict() called {label} {call!r} and then {again!r} on the same "
        "evidence. A verdict that is not a function of its evidence cannot be "
        "defended to anybody")
    assert call == expected, explain(
        key, f"on {label} you called {call!r}, and that is not the call", because)
    grade_reason(reason, evidence, key="m5:verdict:" + key, minimum_keys=2)
    return reason


def grade_alert_verdict(lab):
    """Six situations whose right calls differ, then four one-quantity changes."""
    by_label = {}
    for label, evidence, expected, because in VERDICTS:
        by_label[label] = evidence
        one_verdict(lab, label, evidence, expected, because, f"verdict:{label}")

    for label, field, replacement, expected, because in PERTURBED:
        changed = dict(by_label[label])
        changed[field] = replacement
        one_verdict(lab, f"{label}, with {field} changed to {replacement!r}",
                    changed, expected, because, f"verdict:{label}:{field}")

    # And the same call on the student's own measurements rather than on the
    # check's fixtures, so the verdict is attached to the lab they ran.
    series = load_lab(1).shift_against_fixed()
    own = {"quiet_floor": lab.quiet_floor(series), "observed": float(series[14]),
           "detection_delay_days": lab.detection_delay(series),
           "truth_has_arrived": False, "level": "input"}
    one_verdict(lab, "day 15 of this stream, from your own numbers", own, "ticket",
                "Day 15 is the day the monitor first pages. The inputs have stepped "
                "and nobody has bought a label, so the call is the fourth fixture's "
                "-- a morning's work, not a night's.", "verdict:own")


def body(lab):
    # ---- the quiet floor -----------------------------------------------------
    # Days 1 to 4 are [0.1, 0.9, 0.2, 0.3]. The floor over the first three days
    # is 0.9; an implementation that slices one element short returns 0.1.
    close(lab.quiet_floor([0.1, 0.9, 0.2, 0.3], 3), 0.9, 1e-12,
          "quiet_floor over days 1 to 3 of [0.1, 0.9, 0.2, 0.3]")
    close(lab.quiet_floor([0.1, 0.9, 0.2, 0.3], 1), 0.1, 1e-12,
          "quiet_floor over day 1 alone — day 1 is at position 0")
    close(lab.quiet_floor([0.1, 0.9, 0.2, 0.3], 4), 0.9, 1e-12,
          "quiet_floor over all four days")

    # ---- confirmation and cooldown, on a series the check controls ----------
    series = [0, 0, 1, 1, 1, 0, 1, 1]           # days 1 to 8
    got = list(lab.pages_with_confirmation(series, 0.5, 2, 5))
    assert got == [4], (
        f"pages_with_confirmation on {series} with threshold 0.5, two confirmations "
        f"and a five-day cooldown gave {got}; expected [4]. Day 3 is the first day "
        "over the line, day 4 is the second in a row and pages, and the cooldown "
        "then covers days 5 to 8")

    got = list(lab.pages_with_confirmation(series, 0.5, 2, 2))
    assert got == [4, 8], (
        f"the same series with a two-day cooldown gave {got}; expected [4, 8]. The "
        "cooldown ends on day 6, day 7 starts a fresh run, and day 8 is its second")

    got = list(lab.pages_with_confirmation(series, 0.5, 1, 0))
    assert got == [3, 4, 5, 7, 8], (
        f"one confirmation and no cooldown gave {got}; expected every day over the "
        "line, [3, 4, 5, 7, 8] — that is the naive monitor Lab 1 built")

    # A gap must reset the run. Counting total days over the line rather than
    # consecutive ones is the usual mistake and this is where it shows.
    got = list(lab.pages_with_confirmation([1, 0, 1, 0, 1], 0.5, 2, 0))
    assert got == [], (
        f"a series that is over the line on days 1, 3 and 5 but never twice in a row "
        f"gave {got}; expected no pages at all. A day under the threshold resets the "
        "run to nought — confirmation means consecutive, not cumulative")

    # The first page must not be blocked by a cooldown that has not happened yet.
    got = list(lab.pages_with_confirmation([1, 1, 1], 0.5, 2, 99))
    assert got == [2], (
        f"with a 99-day cooldown the first page still happens on day 2; you gave {got}")

    # ---- the same rule on the real series ------------------------------------
    series = load_lab(1).shift_against_fixed()
    raw = load_lab(1).days_over(series, SHIFT_THRESHOLD)
    paged = list(lab.pages_with_confirmation(series))
    assert len(raw) == 14, f"the check expected 14 raw breaches and found {len(raw)}"
    assert paged == [15, 20, 25], (
        f"on the served days with the shipped settings you paged on {paged}; expected "
        "[15, 20, 25]. Day 14 is the first breach, day 15 confirms it, and the "
        "five-day cooldown then allows day 20 and day 25")
    assert len(paged) < len(raw) / 4, (
        f"{len(paged)} pages against {len(raw)} breaches. The point of the block is "
        "that one change should wake one person once, not fourteen times")

    # ---- what confirmation costs ---------------------------------------------
    close(lab.detection_delay(series, SHIFT_THRESHOLD, 2), 1, 0,
          "detection_delay with two confirmations — the first breach is day 14 and "
          "the first page day 15")
    close(lab.detection_delay(series, SHIFT_THRESHOLD, 1), 0, 0,
          "detection_delay with one confirmation, which is no confirmation at all")
    close(lab.detection_delay(series, SHIFT_THRESHOLD, 5), 4,  0,
          "detection_delay with five confirmations — four days of silence bought")
    assert lab.detection_delay([0.0] * 10, SHIFT_THRESHOLD, 2) == -1, (
        "detection_delay on a series that never crosses should return -1, so a caller "
        "can tell 'no delay' from 'never detected'")

    # The real series crosses once and stays over, so its delay is always the
    # number of confirmations minus one — which a table of three answers gives
    # without measuring anything. Here the run breaks after the first crossing,
    # so the two are no longer the same number.
    broken = [1, 0, 1, 1, 1]                       # over the line on days 1, 3, 4, 5
    close(lab.detection_delay(broken, 0.5, 2), 3, 0,
          "detection_delay on [1, 0, 1, 1, 1] with two confirmations. The first "
          "crossing is day 1, the run resets on day 2, and the first two days in a "
          "row end on day 4 — so three days are lost, not one")
    close(lab.detection_delay(broken, 0.5, 3), 4, 0,
          "the same series with three confirmations: the first three in a row end on "
          "day 5, four days after the first crossing")
    close(lab.detection_delay(broken, 0.5, 1), 0, 0,
          "the same series with one confirmation: the first crossing is the first "
          "page, so nothing is lost")

    # ---- the threshold, from the floor rather than from taste ----------------
    floor = lab.quiet_floor(series)
    close(floor, 0.0573, 0.002, "the quiet floor of the real series over days 1 to 13")
    chosen = lab.lowest_safe_threshold(series)
    assert chosen == 0.10, (
        f"lowest_safe_threshold gave {chosen}; expected 0.10. The floor is "
        f"{floor:.4f}, so 0.05 fires on a quiet day and 0.10 does not")
    assert lab.lowest_safe_threshold(series, 13, [0.5, 1.0]) == 0.5, (
        "with only 0.5 and 1.0 on offer, the smallest safe one is 0.5")

    # A candidate exactly equal to the floor is not safe: days_over fires at or
    # over the line, so a threshold of 0.10 against a quiet day of 0.10 pages.
    tie = lab.lowest_safe_threshold([0.10, 0.02, 0.9, 0.9], 2, [0.05, 0.10, 0.25])
    assert tie == 0.25, (
        f"with a quiet floor of exactly 0.10 you chose {tie}; expected 0.25. A "
        "threshold equal to the floor fires on the quiet day that produced it — the "
        "candidate has to clear the floor, not match it")
    assert lab.lowest_safe_threshold(series, 13, [0.01, 0.02]) is None, (
        "when every candidate fires on a quiet day the honest answer is None, not the "
        "largest candidate. A threshold below the noise floor is a monitor that pages "
        "about nothing, and pretending otherwise is how alerts get switched off")

    # The shipped threshold has to be defensible against that floor.
    assert SHIFT_THRESHOLD > floor * 4, (
        f"the shipped threshold {SHIFT_THRESHOLD} is not comfortably above the "
        f"measured floor {floor:.4f}")

    # ---- the same monitor, pointed at four columns instead of one ------------
    # A graduate will point it at forty. The floor a threshold has to clear is
    # then the worst quiet day of the worst column, and it is not the column the
    # monitor was built for.
    by_column = lab.shift_by_column()
    assert set(by_column) == set(WATCHED_COLUMNS), (
        f"shift_by_column watched {sorted(by_column)}; expected the four columns in "
        f"lab_support.WATCHED_COLUMNS, {WATCHED_COLUMNS}. Watch what the model is "
        "fed, not a convenient subset of it")
    for name, column_series in by_column.items():
        assert len(column_series) == len(series), (
            f"the series for {name!r} is {len(column_series)} long; expected "
            f"{len(series)}, one entry per served day, the same shape Lab 1 returns")

    # One measure, defined once. If the speed series is not Lab 1's, two
    # definitions of "shift" are loose in the module and the floors below are
    # not comparable with block one's.
    for position, (mine, lab1_value) in enumerate(zip(by_column["speed"], series)):
        close(mine, lab1_value, 1e-12, (
            f"shift_by_column's series for 'speed' on day {position + 1} differs from "
            "Lab 1's shift_against_fixed. It is the same measure against the same "
            "reference; call Lab 1's input_shift rather than writing a second one"))

    # Against the check's own arithmetic on a column that has holes in it, so
    # that filling an absent reading rather than dropping it is caught.
    reference = reference_frame()
    for name in ("rssi1", "rssiC"):
        values = reference[name].dropna().to_numpy()
        for number in (1, 13, 14, 27):
            today = day_frame(number)[name].dropna().to_numpy()
            expected = abs(today.mean() - values.mean()) / values.std(ddof=1)
            close(by_column[name][number - 1], expected, 1e-9, (
                f"the shift of {name!r} on day {number}. An absent beacon reading is "
                "not a low reading: drop the missing rows before taking either mean, "
                "or a change in how often a beacon is heard reads as a change in how "
                "strong it is"))

    family = lab.family_quiet_floor(by_column)
    close(family, 0.1507, 0.002, (
        "the family quiet floor over the four watched columns. It is the largest "
        "movement ANY watched column makes on ANY quiet day, not the largest the "
        "speed column makes"))
    assert family > floor * 2, (
        f"the family floor {family:.4f} is not above the single-column floor "
        f"{floor:.4f}. Watching four columns gives four chances a day to be unlucky, "
        "so the floor a threshold has to clear can only go up")

    # The noisiest column is not the one the monitor was built for, which is the
    # whole point: whoever chose the threshold was looking at 'speed'.
    floors = {name: lab.quiet_floor(column_series)
              for name, column_series in by_column.items()}
    assert max(floors, key=floors.get) != "speed", (
        f"the quiet floors by column came out {floors}. On this stream the noisiest "
        "quiet column is a beacon, not the speed the threshold was chosen against")

    # And the quiet-floor machinery, pointed at the family, moves the line.
    safe_one = lab.lowest_safe_threshold(by_column["speed"])
    safe_family = lab.lowest_safe_threshold(
        [max(day) for day in zip(*by_column.values())])
    assert safe_family > safe_one, (
        f"the lowest safe threshold for the family came out {safe_family} against "
        f"{safe_one} for the speed column alone. A threshold justified against one "
        "column's floor is not justified for four")

    # ---- and the arithmetic that follows from watching forty of them ---------
    close(lab.expected_false_alarms(40, 0.05), 2.0, 1e-12, (
        "expected_false_alarms(40, 0.05): forty columns, each firing on a twentieth "
        "of the days when nothing at all is happening. Two alarms a day, from "
        "nothing. This is the number that kills a monitoring channel in its first "
        "month"))
    close(lab.expected_false_alarms(40, 0.05, 30), 60.0, 1e-12,
          "expected_false_alarms(40, 0.05, 30) — the same forty columns over a month")
    close(lab.expected_false_alarms(1, 0.05, 30), 1.5, 1e-12, (
        "expected_false_alarms(1, 0.05, 30) — one column over a month. The days "
        "argument has to do something, or the count is per day whatever you pass"))
    close(lab.expected_false_alarms(40, 0.00125), 0.05, 1e-12, (
        "expected_false_alarms(40, 0.00125) — the same forty columns, with the "
        "per-column level divided by forty. One alarm in twenty quiet days, which is "
        "what the single-column level was buying in the first place"))
    close(lab.expected_false_alarms(4, 0.05), 0.2, 1e-12,
          "expected_false_alarms(4, 0.05) — this module's own four columns")
    assert lab.expected_false_alarms(0, 0.05, 30) == 0, (
        "watching no columns produces no alarms; yours did")

    # ---- and the call the evidence justifies ---------------------------------
    grade_alert_verdict(lab)


run(3, "03_one_alert", "pages_with_confirmation", body,
    requires=[(1, lambda lab: lab.input_shift([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])),
              (1, lambda lab: lab.days_over([0.0, 1.0], 0.5))])
