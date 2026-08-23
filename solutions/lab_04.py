"""Lab 4 — solution. Closing the loop: retrain, gate, promote, roll back."""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from mlflow.exceptions import MlflowException

from lab_support import NotSolved, approved_model, day_frame, load_champion  # noqa: F401
from service import models

LAB = 4

TRAIN_DAYS = (21, 24)
GATE_DAYS = (25, 27)
MARGIN = 0.01
RELEASE_CALLS = ("promote", "hold", "roll back")


def _days(first_day: int, last_day: int) -> pd.DataFrame:
    return pd.concat([day_frame(number) for number in range(first_day, last_day + 1)],
                     ignore_index=True)


def retrain(first_day: int, last_day: int, features=None):
    features = models.FEATURES if features is None else features
    training = _days(first_day, last_day)
    model = models.train(training, features)
    # One retrain, one run. The window is read off the frame rather than passed
    # in, so the record cannot disagree with the rows it describes.
    models.log_run(model, features, training, _days(*GATE_DAYS))
    return model


def ask_champion(request):
    champion = load_champion()
    try:
        return champion.predict(request), "answered"
    except MlflowException:
        # A refusal is information. Catching every exception here would turn a
        # genuine bug into the same shrug as a schema violation.
        return None, "refused"


def passes_gate(champion, champion_features, candidate, candidate_features,
                frame, margin: float = MARGIN) -> bool:
    # Each model is scored on its own columns. Handing a model a feature list it
    # was not trained on is the training-and-serving mismatch from Module 3, and
    # it fails quietly rather than loudly.
    before = models.accuracy(champion, frame, champion_features)
    after = models.accuracy(candidate, frame, candidate_features)
    return bool(after >= before + margin)


def promote(version: str, model, features) -> dict:
    models.save(version, model, features)
    registry = models.load_registry()
    registry["approved"] = version
    if not registry["history"] or registry["history"][-1] != version:
        registry["history"].append(version)
    # Both pointers move together, or the two stores disagree about what is in
    # service -- which is worse than either of them being wrong alone.
    registered = models.version_of(model)
    registry.setdefault("mlflow_version", {})[version] = registered
    models.save_registry(registry)
    models.set_champion(registered)
    return registry


def rollback() -> dict:
    registry = models.load_registry()
    history = registry["history"]
    position = history.index(registry["approved"])
    if position > 0:
        previous = history[position - 1]
        registry["approved"] = previous
        models.save_registry(registry)
        registered = registry.get("mlflow_version", {}).get(previous)
        if registered is not None:
            models.set_champion(registered)
    return registry


def close_the_loop(margin: float = MARGIN) -> dict:
    champion, champion_features = approved_model()
    gate_frame = _days(*GATE_DAYS)
    record = {"champion": models.accuracy(champion, gate_frame, champion_features)}

    fresh = retrain(*TRAIN_DAYS)
    record["fresh_rows"] = models.accuracy(fresh, gate_frame, models.FEATURES)
    if passes_gate(champion, champion_features, fresh, models.FEATURES,
                   gate_frame, margin):
        promote("v2", fresh, models.FEATURES)
        champion, champion_features = fresh, models.FEATURES

    explained = retrain(*TRAIN_DAYS, features=models.EXPLAINED_FEATURES)
    record["with_crew"] = models.accuracy(explained, gate_frame, models.EXPLAINED_FEATURES)
    # Against whatever is in service now -- which is v2 if it was promoted a
    # moment ago. Gating against a model already retired justifies a release
    # with a comparison nobody is running any more.
    if passes_gate(champion, champion_features, explained,
                   models.EXPLAINED_FEATURES, gate_frame, margin):
        promote("v3", explained, models.EXPLAINED_FEATURES)

    record["approved"] = models.load_registry()["approved"]
    return record


def release_verdict(evidence: dict) -> tuple:
    # Read, never edited.
    champion = float(evidence["champion_accuracy"])
    candidate = float(evidence["candidate_accuracy"])
    margin = float(evidence["margin"])
    half_width = float(evidence["wilson_half_width"])
    labels = evidence["gate_labels"]
    on_training_days = bool(evidence["measured_on_training_days"])
    regressed = bool(evidence["champion_regressed"])

    # One. Production first. A released regression is losing accuracy while
    # anybody argues about candidates, and going back is a pointer move.
    if regressed:
        return "roll back", (
            f"the champion regressed: champion accuracy {champion:g} is below the "
            f"version it replaced, so the candidate accuracy {candidate:g} is not the "
            f"question yet. Rolling back is a pointer move that costs seconds, and no "
            f"margin of {margin:g} justifies leaving a known regression in service")

    # Two. Admissibility. A candidate scored on its own training days always
    # wins, so that number cannot be used to justify anything.
    if on_training_days:
        return "hold", (
            f"the candidate accuracy {candidate:g} was measured on training days, so it "
            f"cannot be compared with the champion accuracy {champion:g} however large "
            f"the gap looks; measure both again on {labels:g} gate labels neither model "
            f"saw before anybody argues about a margin of {margin:g}")

    # Three. The bar, and the bar under the bar. A gain smaller than the Wilson
    # half-width is a gain the measurement cannot tell from nought, whatever the
    # stated margin says -- which is why the half-width is a floor under it.
    gain = candidate - champion
    bar = max(margin, half_width)
    if gain >= bar:
        return "promote", (
            f"candidate accuracy {candidate:g} against champion accuracy {champion:g}, "
            f"on {labels:g} gate labels neither model was trained on, and that gain "
            f"clears both the stated margin of {margin:g} and the Wilson half-width of "
            f"{half_width:g} those labels support")

    return "hold", (
        f"candidate accuracy {candidate:g} against champion accuracy {champion:g} is a "
        f"gain this measurement cannot defend: it does not clear both the stated margin "
        f"of {margin:g} and the Wilson half-width of {half_width:g} on the {labels:g} "
        f"gate labels behind it, so buy more labels or bring a better candidate")


if __name__ == "__main__":
    import json

    import plotly.graph_objects as go

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from _narrate import narrator, show_table, save_figure

    say = narrator(LAB)
    say.info("Lab 4 — retrain, gate, promote, roll back, through both registries at "
             "once: the fifty-line one and the platform one")

    # This demonstration really releases a model, in both stores. It puts them
    # back exactly as setup.sh left them, so `make check` after `make demo`
    # grades the same service the slides measured.
    saved = models.snapshot_stores()
    try:
        registry = models.load_registry()
        say.info("before: the pickle registry approves %r, history %s; the champion "
                 "alias names version %s of the registered model %r",
                 registry["approved"], registry["history"],
                 models.champion_version(), models.MODEL_NAME)

        # ---- what the signature buys, run rather than asserted ---------------
        request = models.prepare(day_frame(25))
        answers, status = ask_champion(request)
        say.info("a well-formed request of %d rows: %s, %.3f of them called aboard",
                 len(request), status, float(pd.Series(answers).mean()))

        shuffled = request[list(reversed(models.FEATURES))]
        reordered, status = ask_champion(shuffled)
        agrees = bool((pd.Series(reordered) == pd.Series(answers)).all())
        say.info("the same rows with the columns reversed (%s): %s, and the answers "
                 "are identical to the well-formed request: %s. The signature matched "
                 "by name, not by position", list(shuffled.columns), status, agrees)

        _, status = ask_champion(request.rename(columns={"rssi1": "rssi_one"}))
        say.info("one column renamed rssi1 -> rssi_one: %s before a single row was "
                 "scored", status)
        _, status = ask_champion(request.drop(columns=["rssi2"]))
        say.info("one column missing (rssi2 dropped): %s. That is the answer to 'how "
                 "do I guarantee the columns are not switched'", status)

        # ---- the loop --------------------------------------------------------
        runs_before = models.run_count()
        record = close_the_loop()
        say.info("trained on days %d-%d, gated on days %d-%d — days nothing was "
                 "trained on — at a margin of %.2f accuracy",
                 *TRAIN_DAYS, *GATE_DAYS, MARGIN)
        say.info("the model in service        %.4f", record["champion"])
        say.info("retrained on fresh rows     %.4f  (+%.4f)", record["fresh_rows"],
                 record["fresh_rows"] - record["champion"])
        say.info("plus the explaining variable %.4f  (+%.4f over fresh rows, which is "
                 "what the second gate had to clear)", record["with_crew"],
                 record["with_crew"] - record["fresh_rows"])
        say.info("%d retrain(s), %d new run(s) in the store — one retrain, one run",
                 2, models.run_count() - runs_before)
        say.info("the registry approves %r and the champion alias names version %s",
                 record["approved"], models.champion_version())

        lineage = models.lineage()
        show_table(lineage, "the registry as a table: every version, its window, its "
                            "gate-day accuracy, and which one is champion", logger=say)

        registry = models.load_registry()
        pickle_side = registry["approved"]
        alias_side = models.champion_version()
        say.info("the two stores agree: pickle %r -> mlflow version %s, alias -> "
                 "version %s", pickle_side, registry["mlflow_version"][pickle_side],
                 alias_side)

        promoted_version = alias_side
        after = rollback()
        say.info("rollback: the registry approves %r, the history is unchanged %s, and "
                 "the alias is back on version %s. Release is moving a pointer, so "
                 "going back is moving it again",
                 after["approved"], after["history"], models.champion_version())

        strict_registry = json.dumps(after)
        say.debug("registry after the rollback: %s", strict_registry)

        # ---- the two figures -------------------------------------------------
        names = ["the model<br>in service", "retrained on<br>fresh rows",
                 "retrained, plus the<br>explaining variable"]
        values = [record["champion"], record["fresh_rows"], record["with_crew"]]
        figure = go.Figure()
        figure.add_trace(go.Bar(x=names, y=values, marker_color=["#E07B39", "#2A78D6",
                                                                 "#2A78D6"],
                                text=[f"{v:.3f}" for v in values], textposition="outside"))
        figure.add_hline(y=record["champion"] + MARGIN,
                         line=dict(color="#C0392B", dash="dash", width=1.5),
                         annotation_text=f"the gate: champion + {MARGIN}",
                         annotation_position="top left")
        figure.update_layout(
            title="Three candidates on days 25 to 27, which none of them was trained on",
            xaxis_title="candidate",
            yaxis_title="accuracy on the gate days (share of rows right)",
            yaxis_range=[0, 1.05], showlegend=False)
        save_figure(figure, "candidates_at_the_gate", LAB, logger=say)

        moves = [("v1", "1", 0), ("v2", registry["mlflow_version"].get("v2", "?"), 1),
                 ("v3", registry["mlflow_version"].get("v3", "?"), 2),
                 ("v2", registry["mlflow_version"].get("v2", "?"), 3)]
        timeline = go.Figure()
        timeline.add_trace(go.Scatter(
            x=[step for _, _, step in moves],
            y=[int(version) if version.isdigit() else 0 for _, version, step in moves],
            mode="lines+markers+text",
            text=[f"{name} (version {version})" for name, version, _ in moves],
            textposition="top center",
            line=dict(color="#2A78D6", width=2), marker=dict(size=14, color="#2A78D6"),
            name="where the champion alias points"))
        timeline.update_layout(
            title="The champion alias over one morning: two promotions and a rollback",
            xaxis_title="alias move (in order: release v2, release v3, roll back)",
            yaxis_title="registered version the alias names (version number)",
            xaxis=dict(tickmode="array", tickvals=[0, 1, 2, 3],
                       ticktext=["in service", "promote v2", "promote v3", "rollback"]),
            showlegend=False)
        save_figure(timeline, "alias_timeline", LAB, logger=say)

        # The decision the gate cannot make: the same arithmetic, four different
        # calls, because what settles it is admissibility, an existing
        # regression, and how many labels are behind the numbers.
        # The two half-widths: every row of the gate days, and the 600 rows a
        # team that buys 200 labels a day for three days would actually have.
        gate_rows, bought_rows = len(_days(*GATE_DAYS)), 600
        every_row, bought = 0.0105, 0.0257
        # The fresh-rows candidate scored on its own training days: the number a
        # gate must never use, and here also a candidate whose gain over what is
        # in service is small enough that the label count decides the call.
        own_days = round(models.accuracy(models.train(_days(*TRAIN_DAYS)),
                                         _days(*TRAIN_DAYS)), 4)
        situations = {
            "fresh rows against the model in service": {
                "champion_accuracy": round(record["champion"], 4),
                "candidate_accuracy": round(record["fresh_rows"], 4),
                "margin": MARGIN, "wilson_half_width": every_row,
                "gate_labels": gate_rows,
                "measured_on_training_days": False,
                "champion_regressed": False},
            "the same candidate, scored on its own training days": {
                "champion_accuracy": round(record["champion"], 4),
                "candidate_accuracy": own_days,
                "margin": MARGIN, "wilson_half_width": every_row,
                "gate_labels": gate_rows,
                "measured_on_training_days": True,
                "champion_regressed": False},
            "the explaining variable against fresh rows, at a five-point bar": {
                "champion_accuracy": round(record["fresh_rows"], 4),
                "candidate_accuracy": round(record["with_crew"], 4),
                "margin": 0.05, "wilson_half_width": every_row,
                "gate_labels": gate_rows,
                "measured_on_training_days": False,
                "champion_regressed": False},
            "a gain of a point and a third, on every row of the gate days": {
                "champion_accuracy": round(record["fresh_rows"], 4),
                "candidate_accuracy": own_days,
                "margin": MARGIN, "wilson_half_width": every_row,
                "gate_labels": gate_rows,
                "measured_on_training_days": False,
                "champion_regressed": False},
            "the same gain, on the 600 labels a team could afford to buy": {
                "champion_accuracy": round(record["fresh_rows"], 4),
                "candidate_accuracy": own_days,
                "margin": MARGIN, "wilson_half_width": bought,
                "gate_labels": bought_rows,
                "measured_on_training_days": False,
                "champion_regressed": False},
            "what was released last week is worse than what it replaced": {
                "champion_accuracy": round(record["champion"], 4),
                "candidate_accuracy": round(record["fresh_rows"], 4),
                "margin": MARGIN, "wilson_half_width": every_row,
                "gate_labels": gate_rows,
                "measured_on_training_days": False,
                "champion_regressed": True},
        }
        for label, evidence in situations.items():
            call, reason = release_verdict(evidence)
            say.info("%s -> %s", label, call)
            say.info("    because: %s", reason)
        say.info("rows four and five are the same two accuracies, %.4f against %.4f, "
                 "and the same stated bar of %.2f. What differs is how many labels are "
                 "behind them -- %d rows at a Wilson half-width of %.4f against %d "
                 "bought rows at %.4f -- and the call differs with them, because a "
                 "gain the measurement cannot see is not a gain. That is why a release "
                 "is a decision and not a comparison",
                 own_days, round(record["fresh_rows"], 4), MARGIN,
                 gate_rows, every_row, bought_rows, bought)

        say.info("what the check grades: retrain reproduces service.models.train and "
                 "logs one run with the window as parameters and the gate-day accuracy "
                 "as a metric; ask_champion answers a reordered frame and refuses a "
                 "renamed or missing column; the gate refuses a tie, a worse candidate "
                 "and everything at a margin of 0.95; promote and rollback move both "
                 "registries together; the loop ends at v3 at margin 0.01 and at v2 at "
                 "margin 0.05; and release_verdict calls eight situations and four "
                 "one-quantity changes, with a reason built out of the evidence")
    finally:
        models.restore_stores(saved)
        say.info("both stores restored to exactly what setup.sh wrote: registry %r, "
                 "champion alias on version %s. A demonstration that releases a model "
                 "must leave the service as it found it",
                 models.load_registry()["approved"], models.champion_version())
