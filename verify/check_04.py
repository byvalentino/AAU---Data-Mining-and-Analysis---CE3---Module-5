#!/usr/bin/env python3
"""Check 4 — retrain, gate, promote, roll back, through both registries.

This check writes to both stores, because promotion that does not touch the
registry is not promotion: `service/artefacts/` (the fifty-line registry and its
pickles) and the local MLflow store (`mlruns.db` and `mlartifacts/`, the runs,
the registered versions and the `champion` alias). It puts both back exactly as
it found them before it exits, whether it passes or fails.

Nothing here compares a run identifier. They are random, and two stores holding
the same work have different ones; every assertion below is about content — the
parameters a run recorded, the metric it recorded, and which version an alias
names.

And most of this check is no longer about the stores at all. An earlier version
graded run counts, filter strings and alias look-ups in lockstep across two
registries, which examines whether a student can drive MLflow. Seven of those
twelve assertions are gone. What replaced them is `release_verdict`: twelve
situations, three possible calls, and a reason that has to be built out of the
numbers the student was handed. Driving the tool is worth one mark; knowing
whether to release is the block.
"""
import sys, pathlib, json                                            # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, explain, grade_reason, not_ready   # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    import pandas as pd                                               # noqa: E402
    from lab_support import approved_model, day_frame                 # noqa: E402
    from service import models                                        # noqa: E402
except ImportError as unready:
    not_ready(unready)


RELEASE_CALLS = ("promote", "hold", "roll back")

# The two half-widths this module can actually quote. Gate on every row of days
# 25 to 27 and the interval is narrow; gate on the 600 rows a team buying 200
# labels a day for three days would have, and it is not.
EVERY_ROW, BOUGHT = 0.0105, 0.0257
GATE_ROWS, BOUGHT_ROWS = 3600, 600


def _evidence(champion, candidate, margin=0.01, half_width=EVERY_ROW,
              labels=GATE_ROWS, on_training_days=False, regressed=False):
    return {"champion_accuracy": champion, "candidate_accuracy": candidate,
            "margin": margin, "wilson_half_width": half_width,
            "gate_labels": labels,
            "measured_on_training_days": on_training_days,
            "champion_regressed": regressed}


# Eight situations whose right calls differ, and every accuracy in them is one
# this module measured: 0.6881 for the model in service on the gate days, 0.8831
# for the fresh-rows candidate, 0.9089 with the explaining variable, 0.8962 for
# the fresh-rows candidate scored on its own training days.
RELEASES = (
    ("fresh rows against the model in service",
     _evidence(0.6881, 0.8831), "promote",
     "A gain of nearly twenty points on days neither model was trained on, and it "
     "clears both the stated margin and the half-width the labels support. If this "
     "one does not promote, nothing ever will and the gate is a refusal machine."),

    ("the same candidate, scored on its own training days",
     _evidence(0.6881, 0.8962, on_training_days=True), "hold",
     "The number is inadmissible, not merely optimistic: a candidate scored on days "
     "it was trained on always wins, so the comparison is meaningless however large "
     "the gap. There is nothing here to promote OR to refuse -- measure it again on "
     "days neither model saw."),

    ("the explaining variable against fresh rows, at a five-point bar",
     _evidence(0.8831, 0.9089, margin=0.05), "hold",
     "The candidate is genuinely better, by 2.6 points, and the bar in force is "
     "five. A margin you set in advance and then argue past is not a margin. Hold, "
     "and either lower the bar deliberately or find a better candidate."),

    ("the same pair, at the shipped one-point bar",
     _evidence(0.8831, 0.9089), "promote",
     "The same two models, the same two numbers, a different bar -- and that is the "
     "whole point of writing the bar down before the result arrives."),

    ("a candidate that ties what is in service",
     _evidence(0.8831, 0.8831), "hold",
     "Equal is not better. A gate that promotes a tie promotes noise, and every "
     "release costs somebody an afternoon of risk for nothing."),

    ("what was released last week is worse than what it replaced",
     _evidence(0.6881, 0.8831, regressed=True), "roll back",
     "There is a good candidate on the table and it is not the question. Something "
     "already in service is losing accuracy that the version before it was not, and "
     "going back is a pointer move that takes seconds. Fix production first, then "
     "argue about the candidate."),

    ("a gain of 1.3 points, measured on every row of the gate days",
     _evidence(0.8831, 0.8962), "promote",
     "0.0131 clears the stated margin of 0.01, and on 3,600 rows the Wilson "
     "half-width is 0.0105, so it clears the measurement too."),

    ("the same gain, measured on the 600 labels a team could afford",
     _evidence(0.8831, 0.8962, half_width=BOUGHT, labels=BOUGHT_ROWS), "hold",
     "The identical gain, and now the labels behind it support a half-width of "
     "0.0257. A gain smaller than the half-width is a gain the measurement cannot "
     "tell from nought, whatever the stated bar says -- which is why the slide calls "
     "the half-width the principled margin and the stated one a floor under it."),
)

# One quantity changed, and the call has to change with it.
RELEASES_PERTURBED = (
    ("fresh rows against the model in service", "measured_on_training_days", True,
     "hold",
     "Nothing about the two numbers changed; what changed is where the candidate's "
     "number came from, and that decides whether it may be used at all."),
    ("fresh rows against the model in service", "champion_regressed", True,
     "roll back",
     "The candidate is still excellent. Something already released is losing "
     "accuracy, and that is asked first."),
    ("a candidate that ties what is in service", "candidate_accuracy", 0.9089,
     "promote",
     "The same champion and the same bar, and now a candidate that is genuinely "
     "better by more than the measurement's own width."),
    ("the explaining variable against fresh rows, at a five-point bar", "margin", 0.01,
     "promote",
     "The only thing that moved is the bar, and the bar is a choice somebody makes "
     "and writes down."),
)


def one_release(lab, label, evidence, expected, because, key):
    """Grade one release call and one reason."""
    handed = dict(evidence)
    result = lab.release_verdict(handed)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"release_verdict() returned {result!r} on {label}; it returns the pair "
        "(call, reason)")
    call, reason = result
    assert call in RELEASE_CALLS, (
        f"release_verdict() called {call!r} on {label}. The three calls are "
        f"{', '.join(repr(c) for c in RELEASE_CALLS)}, spelled exactly like that")
    assert handed == evidence, (
        f"release_verdict() changed the evidence dictionary it was handed while "
        f"judging {label}. A verdict reads its evidence; it does not edit it")
    assert call == expected, explain(
        key, f"on {label} you called {call!r}, and that is not the call", because)
    grade_reason(reason, evidence, key="m5:release:" + key, minimum_keys=2)


def grade_release_verdict(lab):
    """Eight situations whose right calls differ, then four one-quantity changes."""
    by_label = {label: evidence for label, evidence, _, _ in RELEASES}
    for label, evidence, expected, because in RELEASES:
        one_release(lab, label, evidence, expected, because, f"release:{label}")
    for label, field, replacement, expected, because in RELEASES_PERTURBED:
        changed = dict(by_label[label])
        changed[field] = replacement
        one_release(lab, f"{label}, with {field} changed to {replacement!r}", changed,
                    expected, because, f"release:{label}:{field}")


def ask(lab, frame, what):
    """Call the lab's ask_champion and turn a crash into something readable.

    A student who predicts with the raw estimator rather than through the alias
    gets a scikit-learn error on the reordered frame, which is a true fact about
    their code and a useless thing to print as a traceback. Naming the cause is
    the whole job of a check.
    """
    try:
        return lab.ask_champion(frame)
    except Exception as failure:
        raise AssertionError(
            f"ask_champion raised {type(failure).__name__} on {what} instead of "
            f"returning (answers, status): {failure}. Load the released model with "
            "load_champion(), which enforces the signature, and catch "
            "MlflowException so that a refusal is reported rather than raised. "
            "Predicting with the raw estimator instead raises a different error "
            "here and answers nonsense elsewhere") from None


def pooled(first, last):
    return pd.concat([day_frame(n) for n in range(first, last + 1)], ignore_index=True)


def run_recording(first_day, last_day, features):
    """The run whose parameters say it trained on that window with those columns.

    Found by content. A run identifier is random, so a check that remembered one
    would be asserting about this machine rather than about the student's work.
    """
    mlflow, _ = models.tracking()
    found = mlflow.search_runs(
        experiment_names=[models.EXPERIMENT],
        filter_string=(f"params.first_day = '{first_day}' and "
                       f"params.last_day = '{last_day}' and "
                       f"params.features = '{','.join(features)}'"),
        order_by=["attributes.start_time DESC"])
    return None if not len(found) else found.iloc[0]


def body(lab):
    # Snapshot and restore live here rather than at module scope, because the
    # harness wraps only what body() does. A temporary directory that could not
    # be made would otherwise be a bare traceback and exit 1 -- the code for
    # "your code is wrong" -- for a fault that is the machine's.
    saved = models.snapshot_stores()
    try:
        grade(lab)
    finally:
        models.restore_stores(saved)


def grade(lab):
    gate_frame = pooled(25, 27)

    # ---- retrain, and the run it has to write --------------------------------
    runs_before = models.run_count()
    fresh = lab.retrain(21, 24)
    expected = models.accuracy(models.train(pooled(21, 24)), gate_frame)
    close(models.accuracy(fresh, gate_frame), expected, 1e-12, (
        "retrain(21, 24) measured on days 25 to 27. Use service.models.train on the "
        "pooled days, so the seed and the forest's shape are the same as everywhere "
        "else and only the data differs"))

    logged = models.run_count() - runs_before
    assert logged == 1, (
        f"retrain(21, 24) added {logged} run(s) to the store; expected exactly one. "
        "Every retrain is a run: call service.models.log_run(model, features, "
        "training_frame, gate_frame) once, and only once, per retrain")

    record = run_recording(21, 24, models.FEATURES)
    assert record is not None, (
        "no run in the store records params first_day=21, last_day=24 and "
        f"features={','.join(models.FEATURES)}. The training window is what makes a "
        "run answerable months later; log_run reads it off the frame you pass it, so "
        "pass the training frame rather than the gate frame")
    close(float(record["metrics.accuracy_gate_days"]),
          models.accuracy(fresh, gate_frame), 1e-9, (
        "the accuracy_gate_days metric on the run retrain(21, 24) logged. Pass the "
        "gate frame to log_run as its fourth argument: a run whose only metric is the "
        "score on its own training days records the number a gate must never use"))
    explained = lab.retrain(21, 24, models.EXPLAINED_FEATURES)
    with_crew = models.accuracy(explained, gate_frame, models.EXPLAINED_FEATURES)
    close(with_crew, 0.9089, 0.005, "retrain with the explaining variable added")
    assert with_crew > models.accuracy(fresh, gate_frame), (
        "the candidate that can see who is driving should beat the one that cannot")
    # ---- what the signature buys, run rather than asserted -------------------
    request = models.prepare(day_frame(25))
    answers, status = ask(lab, request, "a well-formed request")
    assert status == "answered", (
        f"ask_champion returned status {status!r} on a well-formed request of "
        f"{len(request)} rows with the columns {list(request.columns)}. The alias "
        "names a released model; that frame is exactly what it was logged with")
    assert answers is not None and len(answers) == len(request), (
        "ask_champion answered but returned no answers, or the wrong number of them")

    reordered, status = ask(lab, request[list(reversed(models.FEATURES))],
                           "a frame whose columns are in a different order")
    assert status == "answered", (
        "ask_champion refused a frame holding the right columns in a different "
        "order. The signature matches by name: a reordered frame is corrected, not "
        "rejected, and that is the whole reason to serve through it")
    assert list(pd.Series(reordered)) == list(pd.Series(answers)), (
        "a reordered frame was answered, but with different answers. Predicting on "
        "the raw estimator would do that; the released model reorders by name first. "
        "Load it with load_champion() and predict on that object")

    _, status = ask(lab, request.rename(columns={"rssi1": "rssi_one"}),
                    "a frame with one column renamed")
    assert status == "refused", (
        f"a frame with rssi1 renamed to rssi_one gave status {status!r}; expected "
        "'refused'. A renamed column is a different column, and answering it is how "
        "a pipeline serves nonsense with full confidence")
    _, status = ask(lab, request.drop(columns=["rssi2"]), "a frame with one column missing")
    assert status == "refused", (
        f"a frame missing rssi2 gave status {status!r}; expected 'refused'")

    # ---- the gate ------------------------------------------------------------
    champion, champion_features = approved_model()
    assert lab.passes_gate(champion, champion_features, fresh, models.FEATURES,
                           gate_frame), (
        "the gate refused a candidate that is nearly twenty points more accurate "
        "than the model in service. Check that you measure the candidate on the "
        "frame passed in rather than on its own training days")

    # A candidate that is no better must be refused. Retraining on the same stale
    # days reproduces the champion exactly, so the gate has nothing to justify it.
    stale = models.train(pooled(0, 9))
    assert not lab.passes_gate(champion, champion_features, stale, models.FEATURES,
                               gate_frame), (
        "the gate opened for a candidate retrained on the same stale days, which "
        "scores exactly what the champion scores. Equal is not better — a gate that "
        "promotes a tie promotes noise")

    # And one that is worse, which is what retraining on the wrong week gives you.
    worse = models.train(pooled(14, 20))
    assert not lab.passes_gate(champion, champion_features, worse, models.FEATURES,
                               gate_frame), (
        f"the gate opened for a candidate scoring "
        f"{models.accuracy(worse, gate_frame):.4f} against the champion's "
        f"{models.accuracy(champion, gate_frame, champion_features):.4f}")

    # The margin has to do something, or it is decoration.
    assert not lab.passes_gate(champion, champion_features, fresh, models.FEATURES,
                               gate_frame, margin=0.95), (
        "with a margin of 0.95 nothing on earth should pass the gate; yours did, so "
        "the margin argument is being ignored")

    # Each model scored on its own columns.
    assert lab.passes_gate(fresh, models.FEATURES, explained,
                           models.EXPLAINED_FEATURES, gate_frame), (
        "gating the five-feature candidate against the four-feature champion failed. "
        "Pass each model its own feature list — service.models.accuracy takes one")

    # ---- promote and roll back, in both registries ---------------------------
    models.build()                                   # back to a known v1-only state
    first_alias = models.champion_version()
    registry = lab.promote("v9", fresh, models.FEATURES)
    assert registry["history"] == ["v1", "v9"], (
        f"the history reads {registry['history']}; expected ['v1', 'v9'] — release "
        "appends, it does not replace")
    assert (models.ARTEFACTS / "model_v9.pkl").exists(), (
        "promote moved the pointer without writing the artefact. Use "
        "service.models.save first, or the registry names a model that is not there")
    on_disk = json.loads((models.ARTEFACTS / "registry.json").read_text())
    assert on_disk["approved"] == "v9", (
        "the registry you returned says v9 but the file on disk does not. Save it — "
        "a release that only exists in memory is not a release")

    alias_now = models.champion_version()
    assert alias_now == models.version_of(fresh), (
        f"the champion alias still names version {alias_now}, and the model you "
        f"promoted was registered as version {models.version_of(fresh)}. Promotion "
        "moves the alias: service.models.set_champion(service.models.version_of("
        "model)). A release recorded in one store and not the other leaves the two "
        "disagreeing about what is answering requests")
    # The previous model is still on disk, which is what makes going back possible.
    assert (models.ARTEFACTS / "model_v1.pkl").exists(), (
        "model_v1.pkl is gone. Release is moving a pointer, not overwriting a file; "
        "delete the old artefact and there is nothing to roll back to")

    registry = lab.rollback()
    assert registry["approved"] == "v1", (
        f"after a rollback the registry approves {registry['approved']!r}; expected "
        "v1, the version before the approved one")
    assert registry["history"] == ["v1", "v9"], (
        f"the history reads {registry['history']} after the rollback. Leave it alone: "
        "rolling back does not un-release anything, it records a change of mind")
    assert models.champion_version() == first_alias, (
        f"after the rollback the champion alias names version "
        f"{models.champion_version()}; expected {first_alias}, where it pointed "
        "before the release. Rollback is the same move reversed, in both stores")

    registry = lab.rollback()
    assert registry["approved"] == "v1", (
        f"a second rollback from the first entry in the history gave "
        f"{registry['approved']!r}. There is nothing before v1; leave the pointer "
        "where it is rather than falling off the end")

    # ---- the whole loop ------------------------------------------------------
    models.build()
    record = lab.close_the_loop()
    for key in ("champion", "fresh_rows", "with_crew", "approved"):
        assert key in record, f"close_the_loop did not report '{key}'"

    close(record["champion"], 0.6881, 0.005, (
        "the accuracy close_the_loop reports for the model in service, on days 25 to "
        "27. Measure it on the gate days, not on the training days"))
    close(record["fresh_rows"], 0.8831, 0.005, "the fresh-rows candidate")
    close(record["with_crew"], 0.9089, 0.005, "the candidate that adds `crew`")
    assert record["approved"] == "v3", (
        f"the loop finished with {record['approved']!r} approved; expected v3. Both "
        "candidates clear the margin, so both are released, and the last one wins")

    final = models.load_registry()
    assert final["history"] == ["v1", "v2", "v3"], (
        f"the history reads {final['history']}; expected ['v1', 'v2', 'v3'] — every "
        "release recorded, in order")
    # The registry answers "which model answered on day 23?" as a lookup, and the
    # answer has to carry the window it was trained on.
    lineage = models.lineage()
    champion_row = lineage[lineage["champion"]]
    assert int(champion_row.iloc[0]["first_day"]) == 21, (
        "the champion the loop left behind does not record days 21 to 24 as its "
        "training window, so the registry cannot say what the model in service was "
        "trained on")

    # Raise the bar and the second candidate no longer clears it — but only if it
    # is being compared against the first. Against the retired champion it wins by
    # twenty-two points and sails through, which is exactly the mistake.
    models.build()
    strict = lab.close_the_loop(margin=0.05)
    assert strict["approved"] == "v2", (
        f"with a five-point margin the loop finished with {strict['approved']!r} "
        "approved; expected v2. The first candidate beats the model in service by "
        "nineteen points and is released. The second beats *that* by 2.6 points, "
        "which does not clear a five-point bar — so it is refused. Getting v3 here "
        "means the second gate is still comparing against the original champion "
        "rather than against what is now in service")

    # ---- and the decision the gate cannot make -------------------------------
    # passes_gate answers one arithmetic question. Whether to release is a
    # larger one, and the parts of it the gate cannot see are what this grades.
    grade_release_verdict(lab)

    # And the ordering that makes the second promotion mean anything.
    assert record["with_crew"] > record["fresh_rows"] > record["champion"], (
        "fresh rows should beat the stale model, and the explaining variable should "
        f"beat fresh rows. You reported {record['champion']:.4f}, "
        f"{record['fresh_rows']:.4f}, {record['with_crew']:.4f}")


# No `requires`: this lab reaches into service/models.py and the registry, never
# into another lab. Declaring a dependency on Lab 1 here sent a student off to
# finish work that has nothing to do with closing the loop.
run(4, "04_close_the_loop", "close_the_loop", body)
