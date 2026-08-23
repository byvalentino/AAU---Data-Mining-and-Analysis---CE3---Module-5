"""Lab 2 — solution. Three levels of monitoring, and what each one costs."""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lab_support import (NotSolved, SAMPLE_SIZE, WATCHED, approved_model,   # noqa: F401
                         day_frame, load_lab, predictions, reference_frame,
                         wilson_interval)

LAB = 2
SEED = 20200122

MOVED = {"inputs": 0.10, "output": 0.05, "truth": 0.05}
SEGMENT = "crew"
TRUTH_BUDGET = SAMPLE_SIZE * 7
ALLOCATIONS = ("uniform", "stratified")


def output_rate(model, frame) -> float:
    return float(predictions(model, frame).mean())


def buy_truth(model, frame, sample_size: int = SAMPLE_SIZE, seed: int = SEED):
    # Positional picks against a pooled frame need a reset index, or iloc and
    # the label column disagree about which row is which.
    frame = frame.reset_index(drop=True)
    trials = min(sample_size, len(frame))
    picked = np.random.default_rng(seed).choice(len(frame), size=trials, replace=False)
    sample = frame.iloc[picked]
    successes = int((predictions(model, sample) == sample["aboard"]).sum())
    return successes, trials


def sampled_accuracy(model, frame, sample_size: int = SAMPLE_SIZE, seed: int = SEED):
    successes, trials = buy_truth(model, frame, sample_size, seed)
    low, high = wilson_interval(successes, trials)
    return successes / trials, low, high


def levels_over(first_day: int, last_day: int, sample_size: int = SAMPLE_SIZE,
                seed: int = SEED) -> dict:
    model, _ = approved_model()
    days = list(range(first_day, last_day + 1))
    pooled = pd.concat([day_frame(number) for number in days], ignore_index=True)

    input_shift = load_lab(1).input_shift
    shift = input_shift(reference_frame()[WATCHED].to_numpy(), pooled[WATCHED].to_numpy())

    # One day's budget per day in the period: a week of truth costs seven
    # mornings, and that price is the reason the free levels are watched first.
    labels = sample_size * len(days)
    point, low, high = sampled_accuracy(model, pooled, labels, seed)
    return {"inputs": shift, "output": output_rate(model, pooled),
            "truth": point, "low": low, "high": high, "labels": labels}


def segment_accuracy(model, frame, segment: str = SEGMENT,
                     budget: int = TRUTH_BUDGET, seed: int = SEED,
                     allocation: str = "stratified") -> dict:
    frame = frame.reset_index(drop=True)
    groups = sorted(frame[segment].unique())
    bought = {}

    if allocation == "uniform":
        # One draw from the whole period -- exactly buy_truth's draw -- and then
        # a split. Each group's share of the sample is its share of the traffic,
        # which is precisely why the small group ends up with a wide interval.
        trials = min(budget, len(frame))
        picked = np.random.default_rng(seed).choice(len(frame), size=trials,
                                                    replace=False)
        sample = frame.iloc[picked].reset_index(drop=True)
        right = np.asarray(predictions(model, sample) == sample["aboard"])
        for group in groups:
            inside = np.asarray(sample[segment] == group)
            bought[group] = (int(right[inside].sum()), int(inside.sum()))
    elif allocation == "stratified":
        # The same money, divided evenly. Nothing here knows or cares how rare
        # the small group is, which is the whole point of stratifying.
        share = budget // len(groups)
        for group in groups:
            part = frame[frame[segment] == group].reset_index(drop=True)
            bought[group] = buy_truth(model, part, share, seed)
    else:
        raise ValueError(f"allocation must be one of {ALLOCATIONS}, not {allocation!r}")

    measured = {}
    for group, (successes, trials) in bought.items():
        low, high = wilson_interval(successes, trials)
        measured[group] = {"accuracy": successes / trials if trials else float("nan"),
                           "low": low, "high": high, "labels": trials}
    return measured


def which_segments_moved(earlier: dict, later: dict) -> list:
    return sorted(group for group in set(earlier) & set(later)
                  if abs(later[group]["accuracy"] - earlier[group]["accuracy"])
                  >= MOVED["truth"])


def which_levels_moved(earlier: dict, later: dict) -> list:
    return sorted(name for name, tolerance in MOVED.items()
                  if abs(later[name] - earlier[name]) >= tolerance)


if __name__ == "__main__":
    import plotly.graph_objects as go

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from _narrate import narrator, show_table, save_figure
    from service import models

    say = narrator(LAB)
    say.info("Lab 2 — three levels on the same three weeks: two of them free, one of "
             "them the only one that sees the week that costs most")

    model, features = approved_model()
    say.info("the model in service: the registry approves %r, %d input columns %s; the "
             "days are read from data/stream.parquet, generated with seed 20200122",
             models.load_registry()["approved"], len(features), features)

    weeks = {"stable, days 7-13": (7, 13),
             "covariate shift, days 14-20": (14, 20),
             "concept drift, days 21-27": (21, 27)}
    measured = {}
    for name, (first, last) in weeks.items():
        level = levels_over(first, last)
        measured[name] = level
        say.info("%-28s inputs %.3f reference s.d. | output rate %.3f | bought accuracy "
                 "%.3f, 95 per cent interval [%.3f, %.3f] from %s hand-checked rows",
                 name, level["inputs"], level["output"], level["truth"],
                 level["low"], level["high"], f"{level['labels']:,}")

    table = pd.DataFrame([
        {"week": name, "inputs_sd": round(level["inputs"], 4),
         "output_rate": round(level["output"], 4), "bought_accuracy": round(level["truth"], 4),
         "low": round(level["low"], 4), "high": round(level["high"], 4),
         "labels": level["labels"]}
        for name, level in measured.items()])
    show_table(table, "the three levels, week by week (the table this module exists for)",
               logger=say)

    names = list(weeks)
    first_step = which_levels_moved(measured[names[0]], measured[names[1]])
    second_step = which_levels_moved(measured[names[1]], measured[names[2]])
    say.info("stable -> covariate: %s moved. The world changed and everything noticed",
             first_step)
    say.info("covariate -> concept: %s moved. The inputs are in the same place "
             "(%.3f against %.3f) and the model says the same thing (%.3f against "
             "%.3f), while bought accuracy falls %.3f to %.3f",
             second_step, measured[names[1]]["inputs"], measured[names[2]]["inputs"],
             measured[names[1]]["output"], measured[names[2]]["output"],
             measured[names[1]]["truth"], measured[names[2]]["truth"])
    say.info("the two weeks' intervals do not overlap ([%.3f, %.3f] against [%.3f, "
             "%.3f]) — a conservative heuristic that the drop is real, not a test "
             "(Schenker & Gentleman, 2001)",
             measured[names[1]]["low"], measured[names[1]]["high"],
             measured[names[2]]["low"], measured[names[2]]["high"])

    # One purchase of one day rather than seven, to price the interval: the same
    # question, a seventh of the budget, and no longer answerable.
    # The seed is the day, so each purchase is a different morning's work -- and
    # so these are the two numbers the slide prints.
    day_19 = sampled_accuracy(model, day_frame(19), SAMPLE_SIZE, SEED + 19)
    day_24 = sampled_accuracy(model, day_frame(24), SAMPLE_SIZE, SEED + 24)
    say.info("one morning instead of a week — day 19 %.3f [%.3f, %.3f], day 24 %.3f "
             "[%.3f, %.3f]; they overlap, so one day of labels cannot say which change "
             "did it", *day_19, *day_24)

    labels = [name for name in weeks]
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=labels, y=[measured[n]["inputs"] for n in labels], name="inputs (reference s.d.)",
        marker_color="#2A78D6", offsetgroup=0))
    figure.add_trace(go.Bar(
        x=labels, y=[measured[n]["output"] for n in labels], name="output rate (share)",
        marker_color="#52514E", offsetgroup=1))
    figure.add_trace(go.Bar(
        x=labels, y=[measured[n]["truth"] for n in labels], name="bought accuracy (share)",
        marker_color="#E07B39", offsetgroup=2,
        error_y=dict(type="data", symmetric=False,
                     array=[measured[n]["high"] - measured[n]["truth"] for n in labels],
                     arrayminus=[measured[n]["truth"] - measured[n]["low"] for n in labels],
                     color="#C0392B", thickness=2, width=8)))
    figure.update_layout(
        title="Three levels, three weeks — only the bought one sees the third week",
        xaxis_title="week of the served month",
        yaxis_title="level (reference standard deviations, or share of rows)",
        barmode="group", legend=dict(x=0.02, y=0.98))
    save_figure(figure, "three_levels_with_intervals", LAB, logger=say)

    # ---- and the same budget, spent by segment -------------------------------
    say.info("the same %d labels a week, bought by segment: `crew` says whether a "
             "person was driving, and it is not one of the model's inputs, so no "
             "number about it has appeared anywhere in this pipeline", TRUTH_BUDGET)

    pools = {name: pd.concat([day_frame(n) for n in range(first, last + 1)],
                             ignore_index=True)
             for name, (first, last) in weeks.items()}
    by_allocation = {
        allocation: {name: segment_accuracy(model, pool, allocation=allocation)
                     for name, pool in pools.items()}
        for allocation in ALLOCATIONS}

    rows = []
    for allocation, by_week in by_allocation.items():
        for name, by_group in by_week.items():
            for group, entry in by_group.items():
                rows.append({"allocation": allocation, "week": name,
                             "crew": int(group), "labels": entry["labels"],
                             "accuracy": round(entry["accuracy"], 4),
                             "low": round(entry["low"], 4),
                             "high": round(entry["high"], 4),
                             "interval_width": round(entry["high"] - entry["low"], 4)})
    show_table(pd.DataFrame(rows), "bought truth by segment: the same budget, two "
                                   "allocations, three weeks", logger=say)

    stratified = by_allocation["stratified"]
    stable_groups = {int(g): e for g, e in stratified[names[0]].items()}
    say.info("the finding nobody was looking for: in the STABLE week, before anything "
             "changed at all, the model is right %.3f [%.3f, %.3f] of the time when "
             "nobody is driving and %.3f [%.3f, %.3f] when somebody is. It has served "
             "one group far worse since the day it was switched on, and every "
             "aggregate accuracy above averaged the two together",
             stable_groups[0]["accuracy"], stable_groups[0]["low"],
             stable_groups[0]["high"], stable_groups[1]["accuracy"],
             stable_groups[1]["low"], stable_groups[1]["high"])

    say.info("stable -> covariate, by segment: %s moved. The aggregate fell %.3f to "
             "%.3f and NEITHER group got worse -- what moved is the mix, from %.3f to "
             "%.3f of rows with somebody driving, towards the group already served "
             "worse (Simpson, 1951)",
             [int(g) for g in which_segments_moved(stratified[names[0]],
                                                   stratified[names[1]])],
             measured[names[0]]["truth"], measured[names[1]]["truth"],
             float(pools[names[0]][SEGMENT].mean()), float(pools[names[1]][SEGMENT].mean()))
    say.info("covariate -> concept, by segment: %s moved, and only that group: %.3f to "
             "%.3f while the other sits at %.3f. This one is a real loss and it falls "
             "on one group",
             [int(g) for g in which_segments_moved(stratified[names[1]],
                                                   stratified[names[2]])],
             {int(g): e for g, e in stratified[names[1]].items()}[1]["accuracy"],
             {int(g): e for g, e in stratified[names[2]].items()}[1]["accuracy"],
             {int(g): e for g, e in stratified[names[2]].items()}[0]["accuracy"])

    uniform_stable = {int(g): e for g, e in by_allocation["uniform"][names[0]].items()}
    say.info("what the allocation bought: uniformly, the crew group gets %d of the %d "
             "rows and an interval %.4f wide; stratified, %d rows and %.4f wide, for "
             "the same money. The price is the other group's interval, %.4f against "
             "%.4f",
             uniform_stable[1]["labels"], TRUTH_BUDGET,
             uniform_stable[1]["high"] - uniform_stable[1]["low"],
             stable_groups[1]["labels"],
             stable_groups[1]["high"] - stable_groups[1]["low"],
             stable_groups[0]["high"] - stable_groups[0]["low"],
             uniform_stable[0]["high"] - uniform_stable[0]["low"])
    say.info("and it is not only precision: on this budget and this seed the uniform "
             "draw reports %s moved between the stable and the covariate week, which "
             "the stratified draw refuses and which every row of the data refuses too. "
             "A wide interval does not merely say less -- it lets a difference that is "
             "not there look like one",
             [int(g) for g in which_segments_moved(by_allocation["uniform"][names[0]],
                                                   by_allocation["uniform"][names[1]])])

    groups = sorted({int(g) for g in stratified[names[0]]})
    segment_figure = go.Figure()
    for group, colour in zip(groups, ("#2A78D6", "#E07B39")):
        values = [{int(g): e for g, e in stratified[n].items()}[group] for n in names]
        segment_figure.add_trace(go.Bar(
            x=names, y=[v["accuracy"] for v in values],
            name=f"crew = {group}" + (" (a person driving)" if group else " (no driver)"),
            marker_color=colour,
            error_y=dict(type="data", symmetric=False,
                         array=[v["high"] - v["accuracy"] for v in values],
                         arrayminus=[v["accuracy"] - v["low"] for v in values],
                         color="#52514E", thickness=2, width=8)))
    segment_figure.add_trace(go.Scatter(
        x=names, y=[measured[n]["truth"] for n in names], mode="lines+markers",
        name="the aggregate everybody was watching",
        line=dict(color="#C0392B", width=2, dash="dash"), marker=dict(size=10)))
    segment_figure.update_layout(
        title="The average hid a group: accuracy by who was driving, "
              f"{TRUTH_BUDGET:,} labels a week split evenly",
        xaxis_title="week of the served month",
        yaxis_title="accuracy on bought labels (share of rows right)",
        barmode="group", legend=dict(x=0.02, y=0.30), yaxis_range=[0, 1.05])
    save_figure(segment_figure, "accuracy_by_segment", LAB, logger=say)

    say.info("what the check grades: output_rate is the mean of the predictions; "
             "buy_truth draws the check's own rows without replacement; 800 rows give "
             "a narrower interval than 50; a seven-day week costs %d labels; the "
             "second comparison names 'truth' alone; stratified allocation gives each "
             "group %d of the %d labels; and by segment nothing moves between the "
             "first two weeks while only the crew group moves between the last two",
             SAMPLE_SIZE * 7, TRUTH_BUDGET // 2, TRUTH_BUDGET)
