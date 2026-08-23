"""Lab 3 — solution. One change should page one person, once."""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lab_support import (NotSolved, CONFIRMATIONS, COOLDOWN, DAYS,    # noqa: F401
                         SHIFT_THRESHOLD, WATCHED_COLUMNS, day_frame,
                         load_lab, reference_frame)

LAB = 3

LAST_QUIET_DAY = 13
CANDIDATES = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
CALLS = ("page", "ticket", "nothing")


def quiet_floor(series, last_quiet_day: int = LAST_QUIET_DAY) -> float:
    # The series starts at day 1, so day n sits at position n - 1.
    return float(max(series[:last_quiet_day]))


def pages_with_confirmation(series, threshold: float = SHIFT_THRESHOLD,
                            confirmations: int = CONFIRMATIONS,
                            cooldown: int = COOLDOWN) -> list:
    paged, run, quiet_until = [], 0, None
    for position, value in enumerate(series):
        day = position + 1
        run = run + 1 if value >= threshold else 0
        if run >= confirmations and (quiet_until is None or day >= quiet_until):
            paged.append(day)
            quiet_until = day + cooldown
    return paged


def detection_delay(series, threshold: float = SHIFT_THRESHOLD,
                    confirmations: int = CONFIRMATIONS) -> int:
    crossings = [position + 1 for position, value in enumerate(series)
                 if value >= threshold]
    if not crossings:
        return -1
    # The cooldown cannot reach the first page -- there is nothing before it to
    # measure from -- so passing zero here is a statement of that, not a choice.
    pages = pages_with_confirmation(series, threshold, confirmations, cooldown=0)
    if not pages:
        return -1
    return pages[0] - crossings[0]


def lowest_safe_threshold(series, last_quiet_day: int = LAST_QUIET_DAY,
                          candidates: list = None) -> float:
    candidates = sorted(CANDIDATES if candidates is None else candidates)
    floor = quiet_floor(series, last_quiet_day)
    for candidate in candidates:
        if candidate > floor:
            return candidate
    return None


def shift_by_column(days: int = DAYS, columns: list = None) -> dict:
    columns = list(WATCHED_COLUMNS if columns is None else columns)
    input_shift = load_lab(1).input_shift
    reference = reference_frame()
    # The reference is read once per column, not once per day: it is the
    # yardstick, and a yardstick re-read every day is block one's failure again.
    references = {name: reference[name].dropna().to_numpy() for name in columns}
    frames = [day_frame(number) for number in range(1, days)]
    return {name: [input_shift(references[name], frame[name].dropna().to_numpy())
                   for frame in frames]
            for name in columns}


def family_quiet_floor(series_by_column: dict,
                       last_quiet_day: int = LAST_QUIET_DAY) -> float:
    # The alert fires when any watched column fires, so the floor it has to
    # clear is the worst quiet day of the worst column.
    return float(max(quiet_floor(series, last_quiet_day)
                     for series in series_by_column.values()))


def expected_false_alarms(columns: int, level: float, days: int = 1) -> float:
    return float(columns) * float(level) * float(days)


def alert_verdict(evidence: dict) -> tuple:
    # Read, never edited: the caller keeps its own measurements.
    floor = float(evidence["quiet_floor"])
    observed = float(evidence["observed"])
    delay = evidence["detection_delay_days"]
    truth = bool(evidence["truth_has_arrived"])
    level = str(evidence["level"])

    # Question one, and it comes first because it overrules the other two. The
    # floor is the largest movement this monitor produced while nothing at all
    # was happening, so a movement no larger than that is not evidence of
    # anything -- and no threshold that clears the floor could have fired on it.
    if observed <= floor:
        return "nothing", (
            f"the observed movement {observed:g} is at or under the quiet floor "
            f"{floor:g}, which is the largest this monitor reports on a day when "
            f"nothing is happening, so at the {level} level there is nothing here "
            f"to act on")

    # Question two. Only the outcome level says somebody was served worse, and
    # only bought truth establishes it. Both have to hold before anybody is woken.
    if level == "outcome" and truth:
        return "page", (
            f"bought truth has arrived and the observed loss at the outcome level is "
            f"{observed:g} against a quiet floor of {floor:g}; that is a measured cost "
            f"to whoever is being served, it will not repair itself overnight, and "
            f"{delay:g} day(s) of detection delay have already been spent")

    # Question three. Everything else is a morning's work, not a night's. An
    # input or output movement is a reason to buy truth, and nobody can act at
    # three in the morning on a number that may still be noise.
    return "ticket", (
        f"the observed movement {observed:g} clears the quiet floor {floor:g}, so "
        f"something really did move at the {level} level, but "
        + ("no bought truth has arrived yet" if not truth
           else "this is not the outcome level")
        + f", so the honest next step is a morning of buying truth rather than "
          f"waking somebody {delay:g} day(s) after the fact")


if __name__ == "__main__":
    import pandas as pd
    import plotly.graph_objects as go

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from _narrate import narrator, show_table, save_figure

    say = narrator(LAB)
    say.info("Lab 3 — the same correct signal, and the difference between fourteen "
             "true pages and three useful ones")

    lab1 = load_lab(1)
    series = lab1.shift_against_fixed()
    say.info("Lab 1's fixed-reference series: %d days, generated stream, seed 20200122",
             len(series))

    floor = quiet_floor(series)
    say.info("quiet floor over days 1 to %d (nothing happening): %.4f reference "
             "standard deviations — sampling alone moves the monitor this far",
             LAST_QUIET_DAY, floor)
    say.info("shipped threshold %.2f reference s.d., which is %.1f times the measured "
             "floor. Quote the floor whenever you quote the threshold",
             SHIFT_THRESHOLD, SHIFT_THRESHOLD / floor)
    say.info("lowest safe candidate from %s: %s — anything lower fires on a quiet day",
             CANDIDATES, lowest_safe_threshold(series))
    say.info("and when nothing on the ladder clears the floor the answer is %r, which "
             "is a finding rather than a failure: %s",
             lowest_safe_threshold(series, LAST_QUIET_DAY, [0.01, 0.02]),
             "no line can be drawn above the noise")

    raw = lab1.days_over(series, SHIFT_THRESHOLD)
    paged = pages_with_confirmation(series)
    say.info("threshold alone: %d pages on days %s — every one of them true, and all "
             "of them about one event", len(raw), raw)
    say.info("with %d confirmations and a %d-day cooldown: %d pages on days %s",
             CONFIRMATIONS, COOLDOWN, len(paged), paged)
    say.info("what confirmation costs: %d day(s). The first breach is day %d and the "
             "first page day %d — measured, not assumed",
             detection_delay(series), raw[0], paged[0])
    say.info("on a series whose run breaks, [1, 0, 1, 1, 1] at two confirmations, the "
             "delay is %d rather than confirmations − 1 = 1",
             detection_delay([1, 0, 1, 1, 1], 0.5, 2))

    table = pd.DataFrame({
        "day": range(1, len(series) + 1),
        "shift_sd": [round(value, 4) for value in series],
        "over_the_line": [value >= SHIFT_THRESHOLD for value in series],
        "pages": [(position + 1) in paged for position in range(len(series))],
    })
    show_table(table, "every served day: the signal, the breach, and the page",
               logger=say)

    days = list(range(1, len(series) + 1))
    figure = go.Figure()
    # The confirmation window drawn on the picture: the days a run was building
    # but nobody had been woken yet.
    for page in paged:
        figure.add_vrect(x0=page - CONFIRMATIONS + 0.5, x1=page + 0.5,
                         fillcolor="#2A78D6", opacity=0.12, line_width=0,
                         annotation_text="confirmation window",
                         annotation_position="top left", annotation_font_size=10)
    figure.add_hline(y=SHIFT_THRESHOLD, line=dict(color="#52514E", dash="dash", width=1.5),
                     annotation_text=f"threshold {SHIFT_THRESHOLD} reference s.d.",
                     annotation_position="bottom right")
    figure.add_hline(y=floor, line=dict(color="#C0392B", dash="dot", width=1.5),
                     annotation_text=f"measured quiet floor {floor:.4f}",
                     annotation_position="top right")
    figure.add_trace(go.Scatter(x=days, y=series, mode="lines", name="input shift",
                                line=dict(color="#2A78D6", width=2)))
    figure.add_trace(go.Scatter(x=raw, y=[series[d - 1] for d in raw], mode="markers",
                                name=f"threshold alone — {len(raw)} pages",
                                marker=dict(color="#E07B39", size=10)))
    figure.add_trace(go.Scatter(x=paged, y=[series[d - 1] for d in paged], mode="markers",
                                name=f"confirmed and cooled down — {len(paged)} pages",
                                marker=dict(color="#C0392B", size=18, symbol="star")))
    figure.update_layout(
        title="One change, three pages: the confirmation window is shaded",
        xaxis_title="day of the served month (day)",
        yaxis_title="input shift (reference standard deviations)",
        legend=dict(x=0.02, y=0.55))
    save_figure(figure, "alerts_with_confirmation", LAB, logger=say)

    # The verdict, on four situations that differ in exactly one thing each. The
    # point of printing all four is that the call changes while the arithmetic
    # does not: a monitor's output is a measurement, and the decision is a
    # separate act with its own reasoning.
    situations = {
        "a quiet day, nobody has bought a label": {
            "quiet_floor": floor, "observed": series[4],
            "detection_delay_days": detection_delay(series),
            "truth_has_arrived": False, "level": "input"},
        "day 15: the inputs stepped, no truth yet": {
            "quiet_floor": floor, "observed": series[14],
            "detection_delay_days": detection_delay(series),
            "truth_has_arrived": False, "level": "input"},
        "the same movement seen at the model's own output, no truth yet": {
            "quiet_floor": floor, "observed": series[14],
            "detection_delay_days": detection_delay(series),
            "truth_has_arrived": False, "level": "output"},
        "the same movement, at the outcome level, with truth bought": {
            "quiet_floor": floor, "observed": series[14],
            "detection_delay_days": detection_delay(series),
            "truth_has_arrived": True, "level": "outcome"},
    }
    verdicts = []
    for label, evidence in situations.items():
        call, reason = alert_verdict(evidence)
        verdicts.append({"situation": label, "level": evidence["level"],
                         "truth_has_arrived": evidence["truth_has_arrived"],
                         "observed": round(float(evidence["observed"]), 4),
                         "call": call})
        say.info("%s -> %s", label, call)
        say.info("    because: %s", reason)
    show_table(pd.DataFrame(verdicts),
               "the same monitor, four situations, three different calls", logger=say)
    say.info("read the table down the 'call' column: the arithmetic is identical in "
             "the last three rows and the call is not, because the decision rests on "
             "which level moved and on whether anybody has bought truth yet")

    say.info("what the check grades: quiet_floor over days 1 to %d inclusive; pages on "
             "days [15, 20, 25] with the shipped settings; a run that breaks resets; "
             "the delay measured on a broken run; lowest_safe_threshold = 0.10 "
             "here and None when nothing clears the floor; and alert_verdict's call "
             "on six situations whose right answers differ, with a reason built out "
             "of the evidence it was handed", LAST_QUIET_DAY)
