"""Lab 2 — Three levels of monitoring, and what each one costs.

Why this lab exists: two of the three things you can watch cost nothing and are
blind to the failure that costs most, and the only way to believe that is to
measure all three on the same days and read the table across. You prove here
that a week in which the inputs sit still and the model keeps saying the same
thing can still be a week in which the model is ten points less right, and that
saying so honestly needs an interval rather than a decimal point.
Where it sits: Block two — "Three things you can watch", "The table this
module exists for" and "The average hid a group, and the group is who was
driving", and the definition slides "Definition — covariate shift, concept drift
and prior shift", "Definition — the three levels, and what each costs",
"Definition — bought truth and its Wilson interval", "Definition — when a level
has moved, and why overlap is not a test" and "Definition — buying truth by
segment, and what the allocation costs".
What the check grades: output_rate is the mean of the model's own predictions;
buy_truth draws the rows the check draws, without replacement, and never asks
for more rows than the frame holds; a sample of 800 gives a narrower interval
than one of 50; levels_over returns inputs, output, truth, low, high and labels,
buying 200 rows per day so a seven-day week costs 1,400 labels;
which_levels_moved names all three levels between the stable and the covariate
week and only "truth" between the covariate and the concept week;
segment_accuracy spends the same 1,400 labels two ways and gives the smaller
group 700 rows under stratification against about 125 under uniform sampling;
and which_segments_moved returns nothing at all between the stable and the
covariate week and only the crew group between the covariate and the concept
week.
Needs: numpy, pandas, and lab_support for the model, the days and Wilson.

Twenty-five minutes.

Lab 1 watched the inputs. That is the cheapest thing you can watch: it needs no
labels, no waiting, and it runs the moment the data arrives. It is also the
level that is easiest to fool.

There are three levels, and they differ in price and in power:

    the inputs        free            has the world changed?
    the model's own   free            is the model saying different things?
    output
    bought truth      expensive       is the model still right?

The middle level is the one people forget. You do not need labels to notice that
the share of readings called 'aboard' jumped from 55 to 69 per cent. That is the
model telling you something changed, in its own words, at no cost.

The third level needs somebody to sit down and check rows by hand. Two hundred
rows is a morning's work, and the honest output is not a number but an interval —
Module 4's Wilson interval, imported here rather than rewritten, because a share
measured on a sample and reported without an interval is a guess wearing a
decimal point.

Then the part of this lab that is the whole module. Compare the week from day 14
to day 20 against the week from day 21 to day 27. Something serious happens
between them: the model loses another ten points of accuracy. Measure all three
levels on both weeks and find out how many of them notice.

And then the part that decides whether any of it was useful. An accuracy for a
served week is an average over everybody the system served, and an average hides
whoever it is averaged with. So buy the same 1,400 labels a second time, split by
the column that says who was driving, and read the two groups separately. Two
things fall out, and neither of them is on any slide before you measure it: one
of the two weekly drops turns out not to be a drop in accuracy for anybody, and
the model has been serving one group far worse than the other since the first
day it was switched on.

That is also where the sampling decision stops being an abstraction. Draw
uniformly and the smaller group gets whatever share of the sample the traffic
gives it, which here is about a tenth. Split the budget evenly and it gets half.
The same money, a very different sentence about that group, and something given
up on the other one. You measure both and say which you would buy.

What you write: output_rate, buy_truth, sampled_accuracy, levels_over,
which_levels_moved, segment_accuracy and which_segments_moved.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lab_support import (NotSolved, SAMPLE_SIZE, WATCHED, approved_model,   # noqa: E402
                         day_frame, load_lab, predictions, reference_frame,
                         wilson_interval)

LAB = 2
SEED = 20200122

# What counts as a level having moved between two periods. Stated here rather
# than argued about later: a threshold you choose after seeing the numbers is
# not a threshold, it is a decoration. These three are this course's choices,
# the check grades them, and they are on a slide as well as in this file.
MOVED = {"inputs": 0.10,     # reference standard deviations
         "output": 0.05,     # share of readings called aboard
         "truth": 0.05}      # share of predictions that were right

# The column that names the group. `crew` says whether a person was driving.
# The model is not allowed to see it -- it is not one of the four features -- so
# nothing in the pipeline has ever reported a number about it.
SEGMENT = "crew"

# The budget, stated before anything is measured: 1,400 hand-checks for a week,
# which is exactly what levels_over already spends at 200 rows a day for seven
# days. Buying truth by segment is not a request for more money. It is the same
# money, allocated differently, and the allocation is the decision.
TRUTH_BUDGET = SAMPLE_SIZE * 7

# The two ways to spend it.
#   uniform      draw from the whole period and let each group's share of the
#                sample fall where the traffic puts it
#   stratified   divide the budget equally between the groups and draw that many
#                from each, whatever their share of the traffic
ALLOCATIONS = ("uniform", "stratified")


def output_rate(model, frame) -> float:
    """The share of rows the model calls 'aboard'. No labels needed.

    Use predictions(model, frame) from lab_support, which fills absent beacon
    readings the same way training filled them.

    Definition graded by the check:
        r_t = (1/n_t) Σ_i ŷ_{t,i}
        (Breck et al., 2017). Choices: the model's own predictions, never the
        labels; the mean over every row of the frame, pooled days included.
        Slide: "Definition — the three levels, and what each costs".
    Needs: predictions, numpy
    """
    # TODO: predict, take the mean.
    raise NotSolved("output_rate(model, frame) still raises instead of returning a share")


def buy_truth(model, frame, sample_size: int = SAMPLE_SIZE, seed: int = SEED):
    """Hand-check a sample. Return (successes, trials).

    Draw `sample_size` rows without replacement using
    numpy.random.default_rng(seed).choice, predict on exactly those rows, and
    count how many predictions match the 'aboard' column. If the frame holds
    fewer rows than sample_size, check all of them.

    Reset the frame's index first, or positional picks and label lookups will
    disagree once you pass in a pooled multi-day frame.

    Definition graded by the check:
        S ⊂ D drawn without replacement, |S| = min(m, |D|), successes k = Σ_{i ∈ S} 1[ŷ_i = y_i]
        (Wilson, 1927, for what is done with k next). Choices: without
        replacement, so nobody is hand-checked twice and the interval is not
        narrowed by a row counted again; the rows are drawn with
        numpy.random.default_rng(seed).choice and the seed is stated, so the
        check draws the same sample. Slide: "Definition — bought truth and its
        Wilson interval".
    Needs: pandas, numpy, pandas, predictions
    """
    # TODO: sample, predict, count.
    raise NotSolved("buy_truth(model, frame, sample_size, seed) still raises instead "
                    "of returning (successes, trials)")


def sampled_accuracy(model, frame, sample_size: int = SAMPLE_SIZE, seed: int = SEED):
    """Accuracy from a bought sample. Return (point, low, high).

    The point estimate is successes / trials. The bounds come from
    wilson_interval, which you wrote in Module 4 and which lab_support imports
    for you. Report all three: the interval is what stops a good morning from
    being mistaken for a good model.

    Definition graded by the check:
        (k + z²/2)/(n + z²) ± z/(n + z²) · √( k(n − k)/n + z²/4 ), z = 1.96 at 95 per cent
        (Wilson, 1927). Choices: Wilson rather than the normal approximation,
        which collapses to zero width at k = 0 and k = n; z from the standard
        normal at 95 per cent. Slide: "Definition — bought truth and its Wilson
        interval".
    Needs: buy_truth, wilson_interval
    """
    # TODO: buy_truth, then wilson_interval.
    raise NotSolved("sampled_accuracy(model, frame, sample_size, seed) still raises "
                    "instead of returning (point, low, high)")


def levels_over(first_day: int, last_day: int, sample_size: int = SAMPLE_SIZE,
                seed: int = SEED) -> dict:
    """All three levels, measured over the days first_day..last_day inclusive.

    Pool the days into one frame, then return a dict with these keys:

        inputs      the input shift of WATCHED against the reference period,
                    using Lab 1's input_shift — load it with load_lab(1)
        output      the share the model calls aboard, over the pooled frame
        truth       the point estimate from the bought sample
        low, high   the Wilson bounds on that estimate
        labels      how many rows were hand-checked

    Buy `sample_size` rows for every day in the period, so a week costs seven
    times what a day costs. That is what it would cost in real life, and the
    width of the interval is the reason anybody would pay it.

    Definition graded by the check:
        level 1 = Δ_t on the inputs, cost 0 · level 2 = output rate r_t = (1/n_t) Σ_i ŷ_{t,i}, cost 0 · level 3 = accuracy on m bought labels per day, cost m labels
        (Breck et al., 2017, which scores seven monitoring tests, not four).
        Choices: the input level uses Lab 1's fixed reference; the truth level
        buys m rows per day in the period, so the price is linear in the days.
        Slide: "Definition — the three levels, and what each costs".
    Needs: approved_model, day_frame, pandas, load_lab, reference_frame, output_rate,
        sampled_accuracy
    """
    # TODO: concatenate, then the three levels.
    raise NotSolved("levels_over(first_day, last_day) still raises instead of "
                    "returning a dict of three levels")


def segment_accuracy(model, frame, segment: str = SEGMENT,
                     budget: int = TRUTH_BUDGET, seed: int = SEED,
                     allocation: str = "stratified") -> dict:
    """Buy truth **by group**, on a stated budget, and report each group separately.

    `frame` is a period, pooled. `segment` is the column that names the group —
    here `crew`, which says whether a person was driving, and which the model
    has never been allowed to see. `budget` is the total number of rows anybody
    is paying to hand-check over the whole period, not per group.

    Two allocations, and choosing between them is the point of the function:

        "uniform"      draw `budget` rows from the whole frame, exactly as
                       buy_truth does, and then split what you drew by the
                       segment column. Each group's share of the sample is its
                       share of the traffic.
        "stratified"   divide the budget equally between the groups — budget //
                       number of groups each — and draw that many from each
                       group's own rows with buy_truth.

    Return a dict keyed by the distinct values of the segment column, in sorted
    order, each entry a dict with:

        accuracy    successes / labels for that group
        low, high   the Wilson bounds on it
        labels      how many rows of that group were hand-checked

    Then read the two allocations against each other. The same money buys a very
    different statement about the smaller group, and it costs something on the
    larger one. That trade is the decision, and it is worth an argument.

    Definition graded by the check:
        by group g: k_g successes of n_g checked, with Σ_g n_g = budget · uniform: one draw from the period, n_g as the traffic falls · stratified: n_g = budget / |G| for every g, whatever its share of the traffic
        (Neyman, 1934). Choices: the budget is the period's, not the group's, so
        the two allocations are comparable; the same seeded draw as buy_truth, so
        the check draws the same rows; groups in sorted order. Slide: "Definition
        — buying truth by segment, and what the allocation costs".
    Needs: buy_truth, predictions, wilson_interval, numpy, pandas
    """
    # TODO: one draw and a split, or one draw per group. Then Wilson on each.
    raise NotSolved("segment_accuracy(model, frame, segment, budget, seed, allocation) "
                    "still raises instead of returning one entry per group")


def which_segments_moved(earlier: dict, later: dict) -> list:
    """Which groups the model got worse for. Return a sorted list of group values.

    Both arguments come from segment_accuracy. A group moved when the absolute
    difference in its accuracy reaches MOVED["truth"] — the same tolerance
    which_levels_moved uses for bought truth, because it is the same quantity.
    Compare only the groups that appear in both periods.

    This is the function that finds the failure instead of being told it. Run
    this file, compare the stable week against the covariate week, and then the
    covariate week against the concept week, and notice which comparison returns
    an empty list. One of those two aggregate accuracy drops is not a drop in
    accuracy for anybody.

    Definition graded by the check:
        moved(g) ⟺ |accuracy_later(g) − accuracy_earlier(g)| ≥ tolerance, over the groups present in both periods
        (Simpson, 1951). Choice: the same 0.05 tolerance as bought truth at the
        aggregate, so a group is held to the standard the whole is held to; the
        aggregate can move while no group moves, and that is the reading this
        function exists to make available. Slide: "Definition — buying truth by
        segment, and what the allocation costs".
    Needs: MOVED, sorted, abs
    """
    # TODO: one comparison per group, against the tolerance already fixed above.
    raise NotSolved("which_segments_moved(earlier, later) still raises instead of "
                    "returning a list of group values")


def which_levels_moved(earlier: dict, later: dict) -> list:
    """Which levels changed between two periods. Return a sorted list of names.

    Both arguments come from levels_over. A level moved if the absolute
    difference between the two periods reaches its entry in MOVED above. Return
    a sorted list drawn from "inputs", "output" and "truth".

    Run this file. Compare what it prints for the stable week against the
    covariate week, and then for the covariate week against the concept week.
    The second comparison is why this module exists.

    Definition graded by the check:
        moved(level) ⟺ |later − earlier| ≥ tolerance, with tolerances 0.10 reference standard deviations for the inputs, 0.05 for the output rate and 0.05 for bought accuracy
        (Beyer et al., 2016, ch. 6). Choices: the tolerances are this course's,
        fixed in MOVED before any number was seen; "at or over" counts, matching
        Lab 1's breach rule. Slide: "Definition — when a level has moved, and why
        overlap is not a test".

    Definition graded by the check:
        covariate shift: P_ref(X) ≠ P_cur(X) with P(Y | X) unchanged · concept drift: P(Y | X) changes · prior shift: P(Y) changes with P(X | Y) unchanged
        (Moreno-Torres, Raeder, Alaiz-Rodríguez, Chawla & Herrera, 2012; Storkey,
        2009; Lu et al., 2019). This is what the two comparisons tell apart: all
        three levels move between the stable and the covariate week, and only
        bought truth moves between the covariate and the concept week. Slide:
        "Definition — covariate shift, concept drift and prior shift".

    One warning about how the result is read. The slide reports that the two
    weeks' intervals do not overlap. Non-overlapping intervals are a
    *conservative heuristic*, not a hypothesis test: two intervals can overlap
    while the difference is still significant, so the heuristic is a reason to
    look, never a p-value in disguise (Schenker & Gentleman, 2001).
    Needs: MOVED, sorted, abs
    """
    # TODO: compare each of the three against its tolerance.
    raise NotSolved("which_levels_moved(earlier, later) still raises instead of "
                    "returning a list of level names")


if __name__ == "__main__":
    model, _ = approved_model()
    weeks = {"stable, days 7-13": (7, 13),
             "covariate shift, days 14-20": (14, 20),
             "concept drift, days 21-27": (21, 27)}
    measured = {name: levels_over(a, b) for name, (a, b) in weeks.items()}

    print(f"{'week':30}{'inputs':>9}{'output':>9}{'truth':>9}{'95% interval':>22}")
    for name, level in measured.items():
        interval = "[%.3f, %.3f]" % (level["low"], level["high"])
        print(f"{name:30}{level['inputs']:9.3f}{level['output']:9.3f}"
              f"{level['truth']:9.3f}{interval:>22}")

    names = list(weeks)
    print(f"\n{names[0]}  ->  {names[1]}")
    print("  moved:", which_levels_moved(measured[names[0]], measured[names[1]]))
    print(f"{names[1]}  ->  {names[2]}")
    print("  moved:", which_levels_moved(measured[names[1]], measured[names[2]]))
