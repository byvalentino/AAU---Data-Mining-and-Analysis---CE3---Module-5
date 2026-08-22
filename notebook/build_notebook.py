#!/usr/bin/env python3
"""Build Module 5's demonstration notebook, then execute it.

    python "Module 5/notebook/build_notebook.py"

Two sources, and the notebook is explicit about which is which:

    the generated stream   `Module 5/exercises/service/world.py` — twenty-eight
                           days with known regimes, because a monitoring module
                           needs a month and the archive is two days
    the archive            `Module 5/exercises/data/bus_slice.csv.gz` — real
                           telemetry for one shuttle, both days, which identifies
                           nobody. Used at the end to show that the generated
                           covariate shift is the one that actually happened.

The notebook runs from the exercises directory, so a student can open it beside
the labs and every import already works.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
EXERCISES = HERE.parent / "exercises"
OUTPUT = HERE / "Module5_demonstration.ipynb"
MARKDOWN, CODE = "markdown", "code"

CELLS = [
(MARKDOWN, """# Module 5 — Monitoring a model in service

**Data Mining and Analysis (course code CE3) · Aalborg University, Copenhagen**

The model went live on day 10. It is now day 28. Is it still right, and how would
you know?

> **The data, stated once.** Twenty-eight days are *generated* by
> `service/world.py`, because a monitoring module needs a month of days with
> known answers and the archive is two days long. Its parameters come from the
> archive where the archive has them — the crew share from Module 4, the beacon
> rates and the aboard balance from Module 2. How much faster a person drives,
> and what the concept drift does, are modelling choices and are stated as such.
>
> The last section uses the real archive, `data/bus_slice.csv.gz`: shuttle
> VJRD1A10224000055, 22 and 23 January 2020, which identifies nobody.

Run this notebook from the `exercises/` directory. If `service/artefacts/` is
missing, run `bash setup.sh` first.

**Learning objectives.** After working through this notebook you can measure how
far a served day has moved from the days a model trained on; build a three-level
health report and say what each level costs; turn a correct signal into an alert
somebody will still read in six months; and close the loop from a detected
change to a released, reversible fix recorded in a registry.

**Prerequisites.** Modules 1 to 4. Module 3's registry and gate are used
unchanged; Module 4's Wilson interval is imported rather than rewritten.

**Common misconceptions, corrected below.** *"A flat monitoring line means the
model is healthy"* — a week is measured here in which the inputs sat still, the
output rate sat still, and accuracy fell ten points. *"Compare today against
yesterday, it adapts automatically"* — it does, and that is the failure.
*"Retraining fixes drift"* — it recovers most of it; the variable nobody was
watching recovers the rest."""),

(MARKDOWN, """## Hook

A dashboard has shown a flat green line for six weeks and the team believes the
model is healthy.

By the end of this notebook you will have measured a week in which the inputs
did not move, the model's own output did not move, and the model lost ten points
of accuracy."""),

(CODE, '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 110)

# The course palette: reference in blue, current in orange, furniture in grey,
# and red only for what fails or for a line a number has to clear.
BLUE, ORANGE, MUTED, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"

# Figures render inline here and are written beside the notebook as well, so the
# same picture can be dropped into a report without re-running anything.
FIGURES = Path("..") / "notebook" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def show(figure, name, title, x_title, y_title, height=560):
    figure.update_layout(template="plotly_white", title=title, xaxis_title=x_title,
                         yaxis_title=y_title, width=980, height=height,
                         font=dict(size=14), margin=dict(l=70, r=30, t=70, b=60))
    figure.write_image(str(FIGURES / f"{name}.png"), scale=2)
    figure.show()


from service import models
from service.world import stream, reference_period, blind_day, day

# The model the monitor watches: trained on days 0-9, four inputs, fixed seed.
if not (models.ARTEFACTS / "registry.json").exists():
    models.build()

approved = models.load(models.load_registry()["approved"])
model, FEATURES = approved["model"], approved["features"]

reference = reference_period(10)
days = stream(28)

print("features the model sees :", FEATURES)
print("the column it does not  : crew  (the archive's `mode`, renamed)")
print(f"reference period        : days 0-9, {len(reference):,} rows")
print(f"served days             : 1-27, {len(days):,} rows in total")
print(f"accuracy on its own reference period: "
      f"{models.accuracy(model, reference, FEATURES):.4f}")'''),

(MARKDOWN, """## Core concept — three levels, and what each one costs

|  | cost | answers |
|---|---|---|
| the inputs | free | has the world changed? |
| the model's own output | free | is the model saying different things? |
| bought truth | expensive | is the model still right? |

Everything below is those three rows, measured day by day.

> **Definition — input shift, in reference standard deviations.** The distance
> between a served day's mean of a watched column and the reference period's
> mean, divided by the reference period's spread.
>
> `Δ_t = |mean_t − mean_ref| / s_ref, with s_ref the sample standard deviation (ddof = 1) of the reference period`
>
> Glass (1976); Gama et al. (2014), §2.1–2.2. Choices: the reference's spread and
> never the day's own; ddof = 1; the absolute value, because a monitor cares that
> the world moved, not which way.

> **Definition — the model's own output rate.** The share of served rows the
> model calls aboard, counted from the predictions the service already made.
>
> `r_t = (1/n_t) Σ_i ŷ_{t,i}`
>
> Breck et al. (2017), whose rubric scores seven monitoring tests — the three
> levels here are the subset a team with no labelling budget can run."""),

(CODE, '''SHIFT_THRESHOLD = 0.5          # reference standard deviations
WATCHED = "speed"

reference_values = reference[WATCHED].to_numpy()
spread = reference_values.std(ddof=1)


def input_shift(reference_values, current_values):
    """A difference divided by a spread -- and the spread is the REFERENCE's.

    Divide by the current day's instead and the yardstick stretches to fit
    whatever you measure with it. On this stream that turns 0.60 into 0.40,
    which is under any threshold you would have set.
    """
    reference_values = np.asarray(reference_values, dtype=float)
    current_values = np.asarray(current_values, dtype=float)
    return float(abs(current_values.mean() - reference_values.mean())
                 / reference_values.std(ddof=1))


rows = []
for number in range(1, 28):
    today = days[days["day"] == number]
    rows.append({
        "day": number,
        "regime": today["regime"].iloc[0],
        "inputs": input_shift(reference_values, today[WATCHED].to_numpy()),
        "output": float(model.predict(models.prepare(today, FEATURES)).mean()),
        "accuracy": models.accuracy(model, today, FEATURES),
    })

report = pd.DataFrame(rows)
print(report[report["day"].isin([5, 13, 14, 15, 20, 21, 22, 27])].to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))'''),

(MARKDOWN, """Read the last two groups. Day 20 is the end of the covariate week
and day 21 the start of the concept week. The inputs stay where they are, the
model keeps saying the same thing, and accuracy falls off a step.

The three levels have three different scales, so they get three panels rather
than three vertical axes on one."""),

(CODE, '''panels = (("inputs", "input shift<br>(reference s.d.)", "the inputs — free"),
          ("output", "share predicted<br>aboard (share)", "the model's own output — free"),
          ("accuracy", "accuracy<br>(share of rows right)", "truth — expensive"))
figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=[title for _, _, title in panels])
for row, (column, label, _) in enumerate(panels, start=1):
    figure.add_trace(go.Scatter(x=report["day"], y=report[column], mode="lines+markers",
                                line=dict(color=BLUE, width=2), marker=dict(size=6),
                                showlegend=False), row=row, col=1)
    figure.update_yaxes(title_text=label, row=row, col=1)
    for boundary in (14, 21):
        figure.add_vline(x=boundary, line=dict(color=MUTED, width=1, dash="dot"),
                         row=row, col=1)
figure.update_xaxes(title_text="day of the served month (day)", row=3, col=1)
figure.update_layout(template="plotly_white", width=980, height=740,
                     font=dict(size=14), margin=dict(l=95, r=30, t=70, b=60),
                     title="The three levels across twenty-eight days")
figure.write_image(str(FIGURES / "three_levels.png"), scale=2)
figure.show()'''),

(CODE, '''by_regime = report.groupby("regime")[["inputs", "output", "accuracy"]].mean()
print(by_regime.round(3).to_string())
print()

covariate = by_regime.loc["covariate shift"]
concept = by_regime.loc["concept drift"]
print(f"inputs   moved {abs(concept['inputs'] - covariate['inputs']):.3f} "
      "between the two shifted weeks")
print(f"output   moved {abs(concept['output'] - covariate['output']):.3f}")
print(f"accuracy moved {abs(concept['accuracy'] - covariate['accuracy']):.3f}  "
      "<-- the only level that noticed")'''),

(MARKDOWN, """## Worked example — buying truth, with an interval round it

Accuracy above was computed from every label, which in service you do not have.
You buy a sample instead, and a share measured on a sample without an interval
is a guess wearing a decimal point.

> **Definition — bought truth and its Wilson interval.** Draw a sample of served
> rows without replacement, check them by hand, and report the share right as an
> interval rather than as a number.
>
> `S ⊂ D drawn without replacement, |S| = min(m, |D|), successes k = Σ_{i ∈ S} 1[ŷ_i = y_i]`
>
> `(k + z²/2)/(n + z²) ± z/(n + z²) · √( k(n − k)/n + z²/4 ), z = 1.96 at 95 per cent`
>
> Wilson (1927); Brown, Cai & DasGupta (2001). Wilson rather than the normal
> approximation, which collapses to zero width when every checked row is right.
> The interval is Module 4's, imported rather than rewritten."""),

(CODE, '''from lab_support import wilson_interval, SAMPLE_SIZE


def buy_truth(frame, sample_size=SAMPLE_SIZE, seed=20200122):
    """Hand-check a sample. Without replacement -- nobody is checked twice."""
    frame = frame.reset_index(drop=True)
    trials = min(sample_size, len(frame))
    picked = np.random.default_rng(seed).choice(len(frame), size=trials, replace=False)
    sample = frame.iloc[picked]
    right = int((model.predict(models.prepare(sample, FEATURES)) == sample["aboard"]).sum())
    return right, trials


bought = []
for name, (first, last) in {"stable, days 7-13": (7, 13),
                            "covariate, days 14-20": (14, 20),
                            "concept, days 21-27": (21, 27)}.items():
    week = days[(days["day"] >= first) & (days["day"] <= last)]
    right, trials = buy_truth(week, SAMPLE_SIZE * (last - first + 1))
    low, high = wilson_interval(right, trials)
    bought.append({"week": name, "accuracy": right / trials, "low": low, "high": high,
                   "labels": trials})
    print(f"{name:24} {right/trials:.3f}   [{low:.3f}, {high:.3f}]   {trials:,} labels")

bought = pd.DataFrame(bought)
figure = go.Figure(go.Bar(
    x=bought["week"], y=bought["accuracy"], marker_color=[BLUE, ORANGE, RED],
    error_y=dict(type="data", symmetric=False,
                 array=bought["high"] - bought["accuracy"],
                 arrayminus=bought["accuracy"] - bought["low"],
                 color=MUTED, thickness=2, width=10),
    text=[f"{value:.3f}" for value in bought["accuracy"]], textposition="outside"))
figure.update_layout(yaxis_range=[0, 1.05], showlegend=False)
show(figure, "bought_truth", "Bought accuracy, with its 95 per cent Wilson interval",
     "week of the served month", "accuracy (share of hand-checked rows right)")'''),

(CODE, '''# One day's budget instead of a week's -- 200 hand-checks, one morning.
day_19 = buy_truth(days[days["day"] == 19], 200, seed=20200141)
day_24 = buy_truth(days[days["day"] == 24], 200, seed=20200146)
a_low, a_high = wilson_interval(*day_19)
b_low, b_high = wilson_interval(*day_24)

print(f"day 19, covariate : {day_19[0]/200:.3f}   [{a_low:.3f}, {a_high:.3f}]")
print(f"day 24, concept   : {day_24[0]/200:.3f}   [{b_low:.3f}, {b_high:.3f}]")
print(f"\\nintervals overlap : {a_low <= b_high}")
print("one day of labels says the model got worse.")
print("It cannot say which of the two changes did it. A week can.")
print()
print("And note what 'they do not overlap' is worth: it is a conservative")
print("heuristic that the difference is real, not a hypothesis test")
print("(Schenker & Gentleman, 2001).")'''),

(MARKDOWN, """## The average hid a group, and the group is who was driving

Every accuracy above is an average over everybody the system served, and an
average hides whoever it is averaged with. `crew` — whether a person was driving —
is not one of the model's four inputs, so no number about it has appeared
anywhere in this course until now.

Spend the same 1,400 labels a week again, split evenly between the two groups.

> **Definition — buying truth by segment, and what the allocation costs.** The
> same budget divided between the groups whose service you care about, reported
> per group rather than pooled.
>
> `by group g: k_g successes of n_g checked, with Σ_g n_g = budget · uniform: one draw from the period, n_g as the traffic falls · stratified: n_g = budget / |G| for every g, whatever its share of the traffic`
>
> `moved(g) ⟺ |accuracy_later(g) − accuracy_earlier(g)| ≥ tolerance, over the groups present in both periods`
>
> Neyman (1934); Simpson (1951). Equal allocation rather than optimal: the
> optimal rule weights by each group's spread and size, which needs numbers
> nobody has on the first morning."""),

(CODE, '''BUDGET = SAMPLE_SIZE * 7          # 1,400 a week, the same money as above
SEGMENT = "crew"
TOLERANCE = 0.05                  # the same tolerance bought truth is held to


def by_segment(frame, allocation="stratified", budget=BUDGET, seed=20200122):
    """The same budget, two allocations, one entry per group."""
    frame = frame.reset_index(drop=True)
    groups = sorted(frame[SEGMENT].unique())
    counted = {}
    if allocation == "uniform":
        trials = min(budget, len(frame))
        picked = np.random.default_rng(seed).choice(len(frame), size=trials,
                                                    replace=False)
        sample = frame.iloc[picked].reset_index(drop=True)
        right = np.asarray(model.predict(models.prepare(sample, FEATURES))
                           == sample["aboard"])
        for group in groups:
            inside = np.asarray(sample[SEGMENT] == group)
            counted[int(group)] = (int(right[inside].sum()), int(inside.sum()))
    else:
        share = budget // len(groups)
        for group in groups:
            part = frame[frame[SEGMENT] == group].reset_index(drop=True)
            trials = min(share, len(part))
            picked = np.random.default_rng(seed).choice(len(part), size=trials,
                                                        replace=False)
            sample = part.iloc[picked]
            counted[int(group)] = (
                int((model.predict(models.prepare(sample, FEATURES))
                     == sample["aboard"]).sum()), trials)
    out = {}
    for group, (successes, trials) in counted.items():
        low, high = wilson_interval(successes, trials)
        out[group] = {"accuracy": successes / trials, "low": low, "high": high,
                      "labels": trials}
    return out


weeks = {"stable, days 7-13": (7, 13), "covariate, days 14-20": (14, 20),
         "concept, days 21-27": (21, 27)}
pools = {name: days[(days["day"] >= a) & (days["day"] <= b)]
         for name, (a, b) in weeks.items()}
strata = {name: by_segment(pool) for name, pool in pools.items()}

table = pd.DataFrame([
    {"week": name, "crew": group, "labels": entry["labels"],
     "accuracy": round(entry["accuracy"], 4), "low": round(entry["low"], 4),
     "high": round(entry["high"], 4)}
    for name, groups in strata.items() for group, entry in groups.items()])
print(table.to_string(index=False))

stable = strata["stable, days 7-13"]
print()
print("In the STABLE week, before anything changed at all:")
print(f"  nobody driving : {stable[0]['accuracy']:.3f}  "
      f"[{stable[0]['low']:.3f}, {stable[0]['high']:.3f}]")
print(f"  a person driving: {stable[1]['accuracy']:.3f}  "
      f"[{stable[1]['low']:.3f}, {stable[1]['high']:.3f}]")
print("The model has served one group far worse since the day it was switched on,")
print("and every aggregate accuracy above averaged the two together.")'''),

(CODE, '''def which_segments_moved(earlier, later, tolerance=TOLERANCE):
    return sorted(g for g in set(earlier) & set(later)
                  if abs(later[g]["accuracy"] - earlier[g]["accuracy"]) >= tolerance)


names = list(weeks)
mix = {name: float(pool[SEGMENT].mean()) for name, pool in pools.items()}

print("stable -> covariate, by segment :",
      which_segments_moved(strata[names[0]], strata[names[1]]))
print(f"   the aggregate fell, and NEITHER group did. What moved is the mix:")
print(f"   rows with somebody driving {mix[names[0]]:.3f} -> {mix[names[1]]:.3f},")
print("   towards the group the model already served worse (Simpson, 1951).")
print()
print("covariate -> concept, by segment:",
      which_segments_moved(strata[names[1]], strata[names[2]]))
print(f"   one group only: {strata[names[1]][1]['accuracy']:.3f} -> "
      f"{strata[names[2]][1]['accuracy']:.3f}, while the other sits at "
      f"{strata[names[2]][0]['accuracy']:.3f}.")
print("   Two drops that look identical in the aggregate; two opposite events.")

figure = go.Figure()
for group, colour in ((0, BLUE), (1, ORANGE)):
    values = [strata[name][group] for name in names]
    figure.add_trace(go.Bar(
        x=names, y=[v["accuracy"] for v in values],
        name=f"crew = {group}" + (" (a person driving)" if group else " (no driver)"),
        marker_color=colour,
        error_y=dict(type="data", symmetric=False,
                     array=[v["high"] - v["accuracy"] for v in values],
                     arrayminus=[v["accuracy"] - v["low"] for v in values],
                     color=MUTED, thickness=2, width=8)))
figure.update_layout(barmode="group", yaxis_range=[0, 1.05],
                     legend=dict(x=0.02, y=0.30))
show(figure, "accuracy_by_segment",
     "The average hid a group: accuracy by who was driving, 1,400 labels a week",
     "week of the served month", "accuracy on bought labels (share of rows right)")'''),

(CODE, '''# What the allocation bought, and what it cost -- same budget, both ways.
uniform_stable = by_segment(pools["stable, days 7-13"], allocation="uniform")
for group in (0, 1):
    wide = uniform_stable[group]["high"] - uniform_stable[group]["low"]
    tight = stable[group]["high"] - stable[group]["low"]
    print(f"crew={group}:  uniform {uniform_stable[group]['labels']:>4} labels, "
          f"interval {wide:.4f} wide   |   stratified "
          f"{stable[group]['labels']:>4} labels, interval {tight:.4f} wide")
print()
print("Stratifying is not more money. It moves the money, and the price is the")
print("large group's precision. And width is not only precision:")
print("  uniform    stable -> covariate:",
      which_segments_moved(uniform_stable,
                           by_segment(pools["covariate, days 14-20"],
                                      allocation="uniform")))
print("  stratified stable -> covariate:",
      which_segments_moved(strata[names[0]], strata[names[1]]))
print("The uniform draw reports a movement in the small group that the")
print("stratified draw refuses and that every row of the data refuses too.")'''),

(MARKDOWN, """## The textbook blind case

The concept drift above arrives alongside a covariate shift, so the free levels
*do* fire — they simply cannot tell the harmful change from the harmless one.
That is the common case.

Here is the pure form: every feature drawn exactly as on a stable day, every
label inverted."""),

(CODE, '''blind = blind_day()
quiet_floor = report["inputs"][:13].max()          # days 1-13, nothing happening

print(f"input shift vs reference : "
      f"{input_shift(reference_values, blind[WATCHED].to_numpy()):.4f}")
print(f"the measured quiet floor : {quiet_floor:.4f}   (days 1-13)")
print(f"output rate              : "
      f"{model.predict(models.prepare(blind, FEATURES)).mean():.3f}  "
      f"(reference {model.predict(models.prepare(reference, FEATURES)).mean():.3f})")
print(f"accuracy                 : {models.accuracy(model, blind, FEATURES):.4f}")
print()
print("Not 'nothing moved' -- the day is a fresh draw, so everything moved a")
print("little. What is true, and sharper: everything moved by LESS than the")
print("measured quiet floor, so no threshold above the floor can fire on it and")
print("any threshold below the floor fires on every quiet day. There is no line")
print("to draw.")'''),

(MARKDOWN, """## Alerting — one change should page one person, once

The signal is correct. Fourteen of the twenty-seven days are over the line and
every one of them really has shifted. Fourteen pages about one event is how a
monitor gets switched off without anyone deciding to switch it off.

> **Definition — the quiet floor and the lowest safe threshold.**
>
> `floor = max_{1 ≤ t ≤ q} Δ_t over the quiet days, day 1 at position 0`
>
> `τ* = min { τ ∈ C : τ > floor }, and None when no candidate clears the floor`
>
> Beyer et al. (2016), ch. 6; Page (1954). Strictly above the floor, because a
> breach is "at or over" the line.

> **Definition — confirmation, the cooldown, and what they cost.**
>
> `run_t = run_{t−1} + 1 if Δ_t ≥ τ else 0; page at t ⟺ run_t ≥ k and t ≥ p + c, p the last paging day`
>
> `delay = first paging day − first breach, and −1 when the series never crosses`
>
> Page (1954); Beyer et al. (2016), ch. 6. Consecutive, never cumulative."""),

(CODE, '''series = report["inputs"].to_numpy()
raw = [i + 1 for i, v in enumerate(series) if v >= SHIFT_THRESHOLD]


def pages(series, threshold, confirmations=2, cooldown=5):
    paged, run, quiet_until = [], 0, None
    for position, value in enumerate(series):
        day_number = position + 1
        run = run + 1 if value >= threshold else 0
        if run >= confirmations and (quiet_until is None or day_number >= quiet_until):
            paged.append(day_number)
            quiet_until = day_number + cooldown
    return paged


confirmed = pages(series, SHIFT_THRESHOLD)
print(f"quiet floor, days 1-13  : {quiet_floor:.4f}")
print(f"threshold in use        : {SHIFT_THRESHOLD}  "
      f"({SHIFT_THRESHOLD / quiet_floor:.1f} times the floor)")
print(f"threshold alone         : {len(raw)} pages  {raw}")
print(f"confirmation + cooldown : {len(confirmed)} pages  {confirmed}")
print(f"cost of confirmation    : {confirmed[0] - raw[0]} day")

figure = go.Figure()
for page in confirmed:
    figure.add_vrect(x0=page - 1.5, x1=page + 0.5, fillcolor=BLUE, opacity=0.12,
                     line_width=0)
figure.add_hline(y=SHIFT_THRESHOLD, line=dict(color=MUTED, dash="dash", width=1.5),
                 annotation_text=f"threshold {SHIFT_THRESHOLD}",
                 annotation_position="bottom right")
figure.add_hline(y=quiet_floor, line=dict(color=RED, dash="dot", width=1.5),
                 annotation_text=f"measured quiet floor {quiet_floor:.4f}",
                 annotation_position="top right")
figure.add_trace(go.Scatter(x=report["day"], y=series, mode="lines",
                            line=dict(color=BLUE, width=2), name="input shift"))
figure.add_trace(go.Scatter(x=raw, y=[series[d - 1] for d in raw], mode="markers",
                            marker=dict(color=ORANGE, size=11),
                            name=f"threshold alone — {len(raw)} pages"))
figure.add_trace(go.Scatter(x=confirmed, y=[series[d - 1] for d in confirmed],
                            mode="markers", marker=dict(color=RED, size=20, symbol="star"),
                            name=f"confirmed and cooled down — {len(confirmed)}"))
figure.update_layout(legend=dict(x=0.02, y=0.6))
show(figure, "alerts", "One change, three pages: the confirmation window is shaded",
     "day of the served month (day)", "input shift (reference standard deviations)")'''),

(MARKDOWN, """### Forty columns, and two alarms a day from nothing

Everything above watched **one** column. Nobody in production watches one column.
The measure does not change when the monitor watches four; the number of chances
a quiet day gets to be unlucky does.

> **Definition — watching many columns: the family floor and the expected false
> alarms.**
>
> `Δ_{c,t} = |mean_{c,t} − mean_{c,ref}| / s_{c,ref} for each watched column c, missing readings dropped`
>
> `floor_family = max_c max_{1 ≤ t ≤ q} Δ_{c,t}, the largest quiet-day movement any watched column makes`
>
> `E[alarms] = m · α · d for m columns at a per-column level α over d days`
>
> Rabanser, Günnemann & Lipton (2019); Beyer et al. (2016), ch. 6."""),

(CODE, '''WATCHED_COLUMNS = ["speed", "rssi1", "rssi2", "rssiC"]
CANDIDATES = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]

by_column = {}
for name in WATCHED_COLUMNS:
    column_reference = reference[name].dropna().to_numpy()
    by_column[name] = [input_shift(column_reference,
                                   days[days["day"] == n][name].dropna().to_numpy())
                       for n in range(1, 28)]

floors = {name: max(values[:13]) for name, values in by_column.items()}
family_floor = max(floors.values())
for name, value in floors.items():
    print(f"quiet floor, {name:6}: {value:.4f}")
print()
print(f"family floor over {len(WATCHED_COLUMNS)} columns: {family_floor:.4f}  "
      f"({family_floor / floors['speed']:.1f} times the speed column's)")
print(f"the noisiest quiet column is {max(floors, key=floors.get)!r}, "
      "not the column the threshold was chosen against")
print(f"lowest safe candidate, one column : "
      f"{next(c for c in CANDIDATES if c > floors['speed'])}")
print(f"lowest safe candidate, the family : "
      f"{next(c for c in CANDIDATES if c > family_floor)}")

over = sum(1 for values in by_column.values() for value in values[:13]
           if value >= next(c for c in CANDIDATES if c > floors["speed"]))
print()
print(f"held at the one-column line, {over} of "
      f"{len(WATCHED_COLUMNS) * 13} quiet column-days fires")'''),

(CODE, '''# And the arithmetic for the job you will actually have.
def expected_false_alarms(columns, level, days_watched=1):
    return float(columns) * float(level) * float(days_watched)


for columns, level, span in ((1, 0.05, 1), (4, 0.05, 1), (40, 0.05, 1),
                             (40, 0.05, 30), (40, 0.05 / 40, 30)):
    print(f"{columns:3} column(s) at a per-column level of {level:<8.5g} over "
          f"{span:2} day(s): {expected_false_alarms(columns, level, span):.4g} "
          "alarm(s) from nothing at all")
print()
print("Forty columns at a twentieth is two alarms a day and sixty a month,")
print("from nothing happening. Two defences, and they are the same defence:")
print("divide the per-column level by the number of columns, or measure the")
print("family's floor and put the line above that.")'''),

(MARKDOWN, """## Closing the loop

Module 3's registry and gate, unchanged. What changes is the candidate:
retraining on fresh rows is the reflex, adding the variable that explains the
change is the repair.

Everything is measured on days 25 to 27, which nothing was trained on.

> **Definition — the gate, and the margin it opens on.**
>
> `promote ⟺ accuracy(candidate) ≥ accuracy(champion) + margin, both measured on gate days neither model was trained on, each model on its own feature list`
>
> Kreuzberger, Kühl & Hirschl (2023). The margin is 0.01 here, this course's
> choice; Module 3's gate used nought, which promotes a tie."""),

(CODE, '''train_days = days[(days["day"] >= 21) & (days["day"] < 25)]
gate_days = days[days["day"] >= 25]

fresh = models.train(train_days, models.FEATURES)
explained = models.train(train_days, models.EXPLAINED_FEATURES)

result = {
    "the model in service": models.accuracy(model, gate_days, FEATURES),
    "retrained on fresh rows": models.accuracy(fresh, gate_days, models.FEATURES),
    "plus the explaining variable": models.accuracy(
        explained, gate_days, models.EXPLAINED_FEATURES),
    "what it managed before anything changed": models.accuracy(
        model, reference, FEATURES),
}
for name, value in result.items():
    print(f"  {name:42} {value:.4f}")

print(f"\\nand on its OWN training days the fresh-rows candidate scores "
      f"{models.accuracy(fresh, train_days, models.FEATURES):.4f}")
print("which is why a gate measures on days nothing was trained on.")

names = list(result)[:3]
figure = go.Figure(go.Bar(
    x=[name.replace(" ", "<br>", 2) for name in names],
    y=[result[name] for name in names], marker_color=[ORANGE, BLUE, BLUE],
    text=[f"{result[name]:.3f}" for name in names], textposition="outside"))
figure.add_hline(y=result["the model in service"] + 0.01,
                 line=dict(color=RED, dash="dash", width=1.5),
                 annotation_text="the gate: champion + 0.01",
                 annotation_position="top left")
figure.update_layout(yaxis_range=[0, 1.05], showlegend=False)
show(figure, "candidates", "Three candidates on days 25 to 27, which none of them saw",
     "candidate", "accuracy on the gate days (share of rows right)")'''),

(MARKDOWN, """## Every retrain is a run, and release is moving a pointer

The pickle registry above is the fifty-line idea. Beside it sits the same idea
as the industry keeps it: a local MLflow store, `mlruns.db` for the runs and the
registry and `mlartifacts/` for the models. No server and no account — both
files sit next to the exercises.

> **Definition — a run.** One training, recorded once and never edited.
>
> `run = (parameters, data window, metrics, artefact, environment) recorded once and never edited; one retrain, one run`
>
> Zaharia et al. (2018); Pineau et al. (2021).

> **Definition — release by alias, and rollback.**
>
> `release(v): alias champion ↦ v, the artefact never overwritten`
>
> `rollback: alias champion ↦ v_previous, the same move reversed, with the history append-only`
>
> Zaharia et al. (2018); Schelter et al. (2018).

> **Definition — the model signature.**
>
> `signature = (column names, types, order) inferred at logging and enforced at predict: a reorder is corrected, a rename or an absence is refused`
>
> Zaharia et al. (2018). This is the answer to "how do I guarantee the columns
> are not switched": the guarantee is in the artefact, not in the discipline of
> whoever calls it.

The cell below writes to both stores and puts them back exactly as it found
them, so running this notebook does not change what `make check` grades."""),

(CODE, '''saved = models.snapshot_stores()
try:
    print("before        : the registry approves "
          f"{models.load_registry()['approved']!r}, the champion alias names version "
          f"{models.champion_version()}")

    # ---- what the signature buys, run rather than asserted ------------------
    champion = models.load_champion()          # models:/aboard@champion, a lookup
    request = models.prepare(days[days["day"] == 25], FEATURES)
    answers = champion.predict(request)
    print(f"\\na well-formed request of {len(request):,} rows : answered, "
          f"{answers.mean():.3f} called aboard")

    reversed_columns = request[list(reversed(FEATURES))]
    same = champion.predict(reversed_columns)
    print(f"the same rows, columns reversed        : answered, identical answers: "
          f"{bool((same == answers).all())}")

    from mlflow.exceptions import MlflowException
    for label, broken in (("one column renamed", request.rename(columns={"rssi1": "rssi_one"})),
                          ("one column missing", request.drop(columns=["rssi2"]))):
        try:
            champion.predict(broken)
            print(f"{label:39}: ANSWERED -- which would be the bug")
        except MlflowException:
            print(f"{label:39}: refused, before a single row was scored")

    # ---- two retrains, two runs, two promotions and a rollback --------------
    runs_before = models.run_count()
    models.log_run(fresh, models.FEATURES, train_days, gate_days)
    models.set_champion(models.version_of(fresh))
    models.log_run(explained, models.EXPLAINED_FEATURES, train_days, gate_days)
    models.set_champion(models.version_of(explained))
    print(f"\\ntwo retrains added {models.run_count() - runs_before} run(s)")

    print("\\nthe registry as a table:")
    print(models.lineage().to_string(index=False))

    models.set_champion(models.version_of(fresh))
    print(f"\\nafter the rollback the alias names version {models.champion_version()} "
          "-- the same move reversed")
finally:
    models.restore_stores(saved)
    print(f"\\nboth stores restored: registry {models.load_registry()['approved']!r}, "
          f"alias on version {models.champion_version()}")'''),

(MARKDOWN, """## The archive — was any of this real?

The generated covariate shift is a person driving far more often. That is not
invented: it is what happened between 22 and 23 January 2020, measured here on
the real telemetry."""),

(CODE, '''bus = pd.read_csv(Path("data/bus_slice.csv.gz"), low_memory=False)
bus["_t"] = pd.to_datetime(bus["utc_time"], utc=True, format="mixed")
bus["day"] = bus["_t"].dt.date.astype(str)

archive = bus.groupby("day").agg(
    readings=("speed", "size"),
    mean_speed=("speed", "mean"),
    manual_share=("mode", lambda values: float((values == "manual").mean()) * 100),
    mean_payload=("payload", "mean"),
).round(3)

print("shuttle VJRD1A10224000055, all readings, by day\\n")
print(archive.to_string())

first, second = archive.index[0], archive.index[1]
print(f"\\nmanually driven: {archive.loc[first, 'manual_share']:.1f} per cent on "
      f"{first}, {archive.loc[second, 'manual_share']:.1f} per cent on {second}")
print("Nothing in the pipeline was watching that column.")
print("\\nThe slides quote 9.1 and 41.0 for the same thing. Both are correct and")
print("the difference is the grain: Module 4 measured five-minute windows holding")
print("at least 300 readings, this cell measures every reading in the day.")
print("Standing rule 2 -- print the choice beside the number it produced.")'''),

(CODE, '''# Module 4's verdict on the same two days, in one line: one input moved a great
# deal and the target did not. Windows, not readings, because consecutive
# readings correlate at about 0.997 -- Module 1 measured that.
bus["window"] = bus["_t"].dt.floor("5min")
windows = bus.groupby("window").agg(
    mean_speed=("speed", "mean"), mean_payload=("payload", "mean"),
    readings=("speed", "size")).reset_index()
windows = windows[windows["readings"] >= 300]
windows["day"] = windows["window"].dt.date.astype(str)

before = windows[windows["day"] == first]
after = windows[windows["day"] == second]
for column in ("mean_speed", "mean_payload"):
    shift = (after[column].mean() - before[column].mean()) / before[column].std(ddof=1)
    print(f"{column:14} shifted {shift:+.2f} reference standard deviations")
print("\\nThe input moved. The target did not. The right action was none --")
print("which is why 'no material change' has to be a sayable conclusion.")'''),

(MARKDOWN, """## Practice

1. Change the reference in the day-by-day loop from the fixed reference period to
   *yesterday*. Count how many days are over the threshold. **Success:** the
   count drops from 14 to 1, and you can say in one sentence why that is a
   failure rather than an improvement.
2. Raise `SAMPLE_SIZE` from 200 to 800 for the two single-day purchases. **Success:**
   the two intervals no longer overlap, and you can state the label cost of that.
3. Add `crew` to the monitored inputs — not to the model, only to the monitor —
   and compute its input shift day by day. **Success:** it steps on day 14 and is
   flat afterwards, so it would not have caught the concept drift either.
4. Set the gate's margin to 0.05 and re-run the loop, gating each candidate
   against the one now in service. **Success:** the second candidate is refused,
   and you can say what that costs and what it protects against.
5. Simulate the quiet floor properly: draw twenty reference-like days that differ
   only by their random seed and take the largest shift among them. **Success:**
   your number is within a factor of two of 0.057, and you can say why a
   threshold below it is a monitor about nothing.
6. In the registry cell, log a third candidate trained on the *gate* days and
   promote it. **Success:** it scores highest of all and you can say, in one
   sentence, why that number is worthless — and what the registry recorded that
   lets somebody else notice."""),

(MARKDOWN, """## Appendix — why concept drift is invisible by construction

Write the joint distribution of inputs and target as

    P(x, y) = P(x) · P(y | x)

A monitor with no labels can only see samples of `x`. It therefore estimates
`P(x)` and nothing else.

*Covariate shift* is a change in `P(x)` with `P(y | x)` fixed. It is exactly what
the monitor observes, so it is detectable in principle.

*Concept drift* is a change in `P(y | x)` with `P(x)` fixed. Nothing about the
distribution the monitor can see has changed. No binning, no divergence, no
larger sample recovers it, because the information is not present in the sample.

*Prior shift* is the third case in the same taxonomy: `P(y)` changes with
`P(x | y)` fixed (Moreno-Torres et al., 2012; Storkey, 2009).

This is not a limitation of the Population Stability Index or of the Wasserstein
distance. It is a property of the factorisation. The only way to see a change in
`P(y | x)` is to obtain some `y` — which is what "buying truth" means, and why
its price is the operative constraint in monitoring rather than a detail.

Two consequences worth carrying:

- **The output level is not a third source of information.** The model is a
  fixed function of `x`, so its output distribution is a function of `P(x)`. It
  is a cheap and often more useful *view* of the inputs, not an escape from this
  argument.
- **The variable that explains a concept drift is usually an input you did not
  include.** In this module it is `crew`; in the archive it is `mode`. Add it and
  the drift becomes covariate shift in a larger input space — which is
  detectable. That is the deep reason Lab 4's second candidate wins.

One note on binning, because every divergence index needs it and the measure
used above does not. Module 4 measured that at twenty bins most of its index is
the small constant put under an empty bin to keep the logarithm finite, rather
than the data. Ten bins is this course's ceiling, and the shift used throughout
this notebook bins nothing at all."""),

(MARKDOWN, """## References

- Glass, G. V. (1976). *Primary, Secondary, and Meta-Analysis of Research.* Educational Researcher 5(10), 3–8. https://doi.org/10.3102/0013189X005010003
- Page, E. S. (1954). *Continuous Inspection Schemes.* Biometrika 41(1/2), 100–115. https://doi.org/10.1093/biomet/41.1-2.100
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys 46(4). https://doi.org/10.1145/2523813
- Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V. & Herrera, F. (2012). *A unifying view on dataset shift in classification.* Pattern Recognition 45(1), 521–530. https://doi.org/10.1016/j.patcog.2011.06.019
- Storkey, A. (2009). *When Training and Test Sets Are Different*, in Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A. & Lawrence, N., eds., *Dataset Shift in Machine Learning.* MIT Press.
- Lu, J., Liu, A., Dong, F., Gu, F., Gama, J. & Zhang, G. (2019). *Learning under Concept Drift: A Review.* IEEE TKDE 31(12), 2346–2363. https://doi.org/10.1109/TKDE.2018.2876857
- Breck, E., Cai, S., Nielsen, E., Salib, M. & Sculley, D. (2017). *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE Big Data. https://research.google/pubs/pub46555/
- Beyer, B., Jones, C., Petoff, J. & Murphy, N. R. (2016). *Site Reliability Engineering*, chapter 6. O'Reilly. https://sre.google/sre-book/monitoring-distributed-systems/
- Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and Statistical Inference.* JASA 22(158), 209–212 — the interval, via Module 4.
- Schenker, N. & Gentleman, J. F. (2001). *On Judging the Significance of Differences by Examining the Overlap Between Confidence Intervals.* The American Statistician 55(3), 182–186. https://doi.org/10.1198/000313001317097960
- Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F. & Zumar, C. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Engineering Bulletin 41(4), 39–45.
- Schelter, S., Biessmann, F., Januschowski, T., Salinas, D., Seufert, S. & Szarvas, G. (2018). *On Challenges in Machine Learning Model Management.* IEEE Data Engineering Bulletin 41(4), 5–15.
- Kreuzberger, D., Kühl, N. & Hirschl, S. (2023). *MLOps: Overview, Definition, and Architecture.* IEEE Access 11, 31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138
- Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E. & Larochelle, H. (2021). *Improving Reproducibility in Machine Learning Research.* JMLR 22(164), 1–20.
- Perdomo, J., Zrnic, T., Mendler-Dünner, C. & Hardt, M. (2020). *Performative Prediction.* ICML, PMLR 119, 7599–7609.
- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 28.
- Rabanser, S., Günnemann, S. & Lipton, Z. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS 32. https://arxiv.org/abs/1810.11953 — the multiplicity correction when one test is run per column.
- Neyman, J. (1934). *On the Two Different Aspects of the Representative Method: The Method of Stratified Sampling and the Method of Purposive Selection.* Journal of the Royal Statistical Society 97(4), 558–625. https://doi.org/10.2307/2342192
- Simpson, E. H. (1951). *The Interpretation of Interaction in Contingency Tables.* Journal of the Royal Statistical Society, Series B 13(2), 238–241. https://doi.org/10.1111/j.2517-6161.1951.tb00088.x — taught in Module 2; used here on the segment table.
- European Union (2024). *Regulation (EU) 2024/1689, the Artificial Intelligence Act*, Article 72, post-market monitoring by providers, and the post-market monitoring plan; Annex III; Annex IV. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- European Union (2026). *Regulation (EU) 2026/1744, the Digital Omnibus on artificial intelligence*, amending Regulation (EU) 2024/1689: the Annex III obligations apply from 2 December 2027. https://eur-lex.europa.eu/eli/reg/2026/1744/oj

*Generated data: `service/world.py`, calibrated from the archive as described at
the top. Archive: SafeMobility shuttle telemetry, vehicle VJRD1A10224000055,
22–23 January 2020 — vehicle data only, no personal data. All output above is
Author's own, computed by this notebook.*"""),
]


def main(*arguments):
    notebook = new_notebook(cells=[
        new_markdown_cell(text) if kind == MARKDOWN else new_code_cell(text)
        for kind, text in CELLS])
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"}})

    if "--no-run" not in arguments:
        from nbclient import NotebookClient
        # Executed from exercises/, so `from service import models` works exactly
        # as it does in the labs sitting next to it.
        NotebookClient(notebook, timeout=600,
                       resources={"metadata": {"path": str(EXERCISES)}}).execute()

    OUTPUT.write_text(nbformat.writes(notebook))
    executed = sum(1 for cell in notebook.cells if cell.get("outputs"))
    print(f"wrote {OUTPUT.name} — {len(CELLS)} cells, {executed} with output")


if __name__ == "__main__":
    main(*sys.argv[1:])
