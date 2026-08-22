#!/usr/bin/env python3
"""Check 2 — the three levels, and which of them notices concept drift."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready                            # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    import numpy as np                                                # noqa: E402
    import pandas as pd                                               # noqa: E402
    from lab_support import (SAMPLE_SIZE, approved_model, day_frame,  # noqa: E402
                             predictions, wilson_interval)
    from service import models                                        # noqa: E402
except ImportError as unready:
    not_ready(unready)


def body(lab):
    # Lab 4 promotes a new model and leaves it approved. This lab is about
    # watching the model that went into service on day 10, so put the registry
    # back to v1 before measuring anything -- otherwise finishing Lab 4 quietly
    # changes what Lab 2's check is measuring, and the failure it reports is
    # about the wrong thing.
    #
    # Inside body() rather than at module scope: the harness wraps only what
    # body() does, so training here at import time would report a failure to
    # train as exit 1, which is the code for "your code is wrong".
    models.build()

    model, _ = approved_model()

    # ---- the free output level ----------------------------------------------
    stable, shifted = day_frame(5), day_frame(19)
    for frame, name in ((stable, "day 5"), (shifted, "day 19")):
        expected = float(predictions(model, frame).mean())
        close(lab.output_rate(model, frame), expected, 1e-12,
              f"output_rate on {name}")
    assert lab.output_rate(model, shifted) > lab.output_rate(model, stable) + 0.05, (
        "the model calls a much larger share of rows 'aboard' once the world "
        "shifts. Your output_rate reports almost no difference, so it is probably "
        "returning the share of the *labels* rather than of the predictions")

    # ---- bought truth --------------------------------------------------------
    successes, trials = lab.buy_truth(model, stable, 200, 20200122)
    assert trials == 200, f"buy_truth asked for 200 rows and reported {trials} trials"
    frame = stable.reset_index(drop=True)
    picked = np.random.default_rng(20200122).choice(len(frame), size=200, replace=False)
    sample = frame.iloc[picked]
    expected = int((predictions(model, sample) == sample["aboard"]).sum())
    close(successes, expected, 0, (
        "buy_truth on day 5 with seed 20200122. Use "
        "numpy.random.default_rng(seed).choice(len(frame), size, replace=False) "
        "on the reset-index frame, so the check and the lab draw the same rows"))

    # Sampling without replacement, so nobody is hand-checked twice.
    everything, trials = lab.buy_truth(model, stable, len(stable), 7)
    assert trials == len(stable), "buy_truth should check every row when asked for all"
    close(everything / trials, float((predictions(model, stable) == stable["aboard"]).mean()),
          1e-12, "buy_truth over the whole day should equal the day's accuracy")

    small, trials = lab.buy_truth(model, stable.head(30), 200, 3)
    assert trials == 30, (
        f"buy_truth was asked for 200 rows from a 30-row frame and reported {trials} "
        "trials. Check them all rather than asking for more rows than exist")

    # ---- the interval --------------------------------------------------------
    point, low, high = lab.sampled_accuracy(model, stable, 200, 20200122)
    close(point, expected / 200, 1e-12, "sampled_accuracy point estimate")
    reference_low, reference_high = wilson_interval(expected, 200)
    close(low, reference_low, 1e-9, "sampled_accuracy lower bound")
    close(high, reference_high, 1e-9, "sampled_accuracy upper bound")
    assert low < point < high, "the point estimate must sit inside its own interval"

    wide = lab.sampled_accuracy(model, stable, 50, 11)
    narrow = lab.sampled_accuracy(model, stable, 800, 11)
    assert (wide[2] - wide[1]) > (narrow[2] - narrow[1]), (
        "checking 800 rows must give a narrower interval than checking 50. Yours "
        "did not, so the interval is not being built from the number of rows "
        "actually checked")

    # ---- the three levels over a period --------------------------------------
    weeks = {"stable": (7, 13), "covariate": (14, 20), "concept": (21, 27)}
    measured = {name: lab.levels_over(a, b) for name, (a, b) in weeks.items()}
    for name, level in measured.items():
        for key in ("inputs", "output", "truth", "low", "high", "labels"):
            assert key in level, f"levels_over for the {name} week is missing '{key}'"
        assert level["labels"] == SAMPLE_SIZE * 7, (
            f"the {name} week covers seven days, so at {SAMPLE_SIZE} rows a day it "
            f"should cost {SAMPLE_SIZE * 7} labels; yours reported {level['labels']}")

    # Against the check's own pooling, so the grain is not left to chance.
    for name, (first, last) in weeks.items():
        pooled = pd.concat([day_frame(n) for n in range(first, last + 1)],
                           ignore_index=True)
        close(measured[name]["output"], float(predictions(model, pooled).mean()), 1e-9,
              f"the output level over the {name} week")

    close(measured["stable"]["inputs"], 0.0084, 0.002, "the input level, stable week")
    close(measured["covariate"]["inputs"], 0.6148, 0.01, "the input level, covariate week")
    close(measured["concept"]["inputs"], 0.6032, 0.01, "the input level, concept week")

    # ---- the point of the lab ------------------------------------------------
    first = lab.which_levels_moved(measured["stable"], measured["covariate"])
    assert sorted(first) == ["inputs", "output", "truth"], (
        f"between the stable week and the covariate week you found {sorted(first)} "
        "moved. All three should: the inputs step, the model changes what it says, "
        "and it gets less accurate. A monitor that misses this is watching nothing")

    second = lab.which_levels_moved(measured["covariate"], measured["concept"])
    assert sorted(second) == ["truth"], (
        f"between the covariate week and the concept week you found {sorted(second)} "
        "moved. Only 'truth' should. The inputs are distributed the same way in "
        f"both weeks ({measured['covariate']['inputs']:.3f} against "
        f"{measured['concept']['inputs']:.3f}) and the model says nearly the same "
        f"thing ({measured['covariate']['output']:.3f} against "
        f"{measured['concept']['output']:.3f}), while accuracy falls from "
        f"{measured['covariate']['truth']:.3f} to {measured['concept']['truth']:.3f}. "
        "That gap is the reason this module exists")

    # ---- the same money, spent by segment ------------------------------------
    # An accuracy for a served week is an average over everybody the system
    # served, and an average hides whoever it is averaged with. The column that
    # says who was driving is not one of the model's inputs, so nothing in the
    # pipeline has ever reported a number about it.
    frames = {name: pd.concat([day_frame(n) for n in range(first, last + 1)],
                              ignore_index=True)
              for name, (first, last) in weeks.items()}
    stratified = {name: lab.segment_accuracy(model, frame, allocation="stratified")
                  for name, frame in frames.items()}
    uniform = {name: lab.segment_accuracy(model, frame, allocation="uniform")
               for name, frame in frames.items()}

    for name, measured_by_group in stratified.items():
        keyed = {int(group): entry for group, entry in measured_by_group.items()}
        assert sorted(keyed) == [0, 1], (
            f"segment_accuracy on the {name} week reported groups "
            f"{sorted(measured_by_group)}; expected the two values of the 'crew' "
            "column, 0 and 1")
        for group, entry in keyed.items():
            for key in ("accuracy", "low", "high", "labels"):
                assert key in entry, (
                    f"segment_accuracy's entry for crew={group} in the {name} week is "
                    f"missing '{key}'")
            assert entry["labels"] == SAMPLE_SIZE * 7 // 2, (
                f"under stratified allocation each of the two groups gets half the "
                f"{SAMPLE_SIZE * 7}-label budget, which is {SAMPLE_SIZE * 7 // 2}; the "
                f"{name} week's crew={group} got {entry['labels']}. The budget is the "
                "period's, not the group's -- stratifying is not a request for more "
                "money")

    # The fairness finding, which is a finding rather than a claim: the model has
    # served one group far worse since the first day it was switched on, and the
    # aggregate accuracy on every slide before this one never said so.
    stable_by_group = {int(g): e for g, e in stratified["stable"].items()}
    close(stable_by_group[0]["accuracy"], 0.9100, 0.01,
          "the stable week's accuracy for crew=0, on 700 hand-checked rows")
    close(stable_by_group[1]["accuracy"], 0.6457, 0.01,
          "the stable week's accuracy for crew=1, on 700 hand-checked rows")
    assert stable_by_group[1]["high"] < stable_by_group[0]["low"], (
        f"in the stable week -- before anything changed at all -- the two groups' "
        f"intervals overlap: crew=0 [{stable_by_group[0]['low']:.3f}, "
        f"{stable_by_group[0]['high']:.3f}] against crew=1 "
        f"[{stable_by_group[1]['low']:.3f}, {stable_by_group[1]['high']:.3f}]. They "
        "should not: the model serves the two groups very differently and always has")

    # What the allocation buys, at the same price. The small group is about a
    # twelfth of the stable week's traffic, so a uniform draw hands it a twelfth
    # of the sample and an interval twice as wide.
    stable_uniform = {int(g): e for g, e in uniform["stable"].items()}
    assert sum(e["labels"] for e in stable_uniform.values()) == SAMPLE_SIZE * 7, (
        "under uniform allocation the whole budget is drawn from the period and then "
        f"split, so the labels have to add up to {SAMPLE_SIZE * 7}; yours added to "
        f"{sum(e['labels'] for e in stable_uniform.values())}")
    assert stable_uniform[1]["labels"] < stable_by_group[1]["labels"] / 4, (
        f"a uniform draw over the stable week gave the crew group "
        f"{stable_uniform[1]['labels']} rows against stratification's "
        f"{stable_by_group[1]['labels']}. It is about a twelfth of that week's "
        "traffic, so a uniform draw gives it about a twelfth of the sample")
    uniform_width = stable_uniform[1]["high"] - stable_uniform[1]["low"]
    stratified_width = stable_by_group[1]["high"] - stable_by_group[1]["low"]
    assert uniform_width > 2 * stratified_width, (
        f"the crew group's interval is {uniform_width:.4f} wide under uniform "
        f"allocation and {stratified_width:.4f} under stratification, for the same "
        f"{SAMPLE_SIZE * 7} labels. That gap is the whole argument for stratifying, "
        "and yours does not show it")
    assert (stable_uniform[0]["high"] - stable_uniform[0]["low"]
            < stable_by_group[0]["high"] - stable_by_group[0]["low"]), (
        "stratification has to cost something, and it costs precision on the large "
        "group: its interval is wider under stratification than under a uniform draw. "
        "Yours is not, so the budget is not being moved from one group to the other")

    # ---- and the reading that the aggregate table cannot give -----------------
    first_by_segment = [int(group) for group in
                        lab.which_segments_moved(stratified["stable"],
                                                 stratified["covariate"])]
    assert first_by_segment == [], (
        f"between the stable week and the covariate week you found groups "
        f"{first_by_segment} moved. Neither group did: crew=0 goes "
        f"{stable_by_group[0]['accuracy']:.3f} to "
        f"{float(dict((int(g), e) for g, e in stratified['covariate'].items())[0]['accuracy']):.3f} "
        f"and crew=1 goes {stable_by_group[1]['accuracy']:.3f} to "
        f"{float(dict((int(g), e) for g, e in stratified['covariate'].items())[1]['accuracy']):.3f}. "
        "The aggregate fell because the traffic moved towards the group the model "
        "already served worse, not because anybody was served worse than before")

    second_by_segment = [int(group) for group in
                         lab.which_segments_moved(stratified["covariate"],
                                                  stratified["concept"])]
    assert second_by_segment == [1], (
        f"between the covariate week and the concept week you found groups "
        f"{second_by_segment} moved; expected [1] and nothing else. That week is a "
        "real loss and it falls entirely on one group -- which is what makes it a "
        "different event from the week before it")

    # And the allocation is not a detail: at this budget and this seed the uniform
    # draw reports a movement in the small group that neither the stratified draw
    # nor every row of the data agrees with.
    uniform_first = [int(group) for group in
                     lab.which_segments_moved(uniform["stable"], uniform["covariate"])]
    assert uniform_first == [1], (
        f"spending the same budget uniformly gave {uniform_first} between the stable "
        "and the covariate week. On this seed it gives [1] -- a movement in the small "
        "group that the stratified draw refuses and that the whole population "
        "refuses. That is what a 125-row sample buys, and it is the argument")
    population = {}
    for group in (0, 1):
        for name in ("stable", "covariate"):
            part = frames[name][frames[name]["crew"] == group]
            population[(name, group)] = float(
                (predictions(model, part) == part["aboard"]).mean())
    assert abs(population[("covariate", 1)] - population[("stable", 1)]) < 0.05, (
        "measured on every row rather than on a sample, the crew group's accuracy "
        f"moves {population[('stable', 1)]:.4f} to {population[('covariate', 1)]:.4f} "
        "between those two weeks -- inside the tolerance. The uniform sample said "
        "otherwise, and the uniform sample was wrong")

    # And the interval has to be tight enough to carry the claim.
    assert measured["concept"]["high"] < measured["covariate"]["low"], (
        "the two weeks' accuracy intervals overlap, so the drop is not established. "
        f"Covariate [{measured['covariate']['low']:.3f}, "
        f"{measured['covariate']['high']:.3f}], concept "
        f"[{measured['concept']['low']:.3f}, {measured['concept']['high']:.3f}]. "
        "Check that levels_over buys one sample per day in the period")


run(2, "02_three_levels", "levels_over", body,
    requires=[(1, lambda lab: lab.input_shift([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))])
