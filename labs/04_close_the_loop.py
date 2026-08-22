"""Lab 4 — Closing the loop: retrain, gate, promote, and be able to go back.

Why this lab exists: a monitor nobody can act on is a decoration, and the action
— retrain, measure, release, and be able to undo the release — is the only part
of the loop that touches production. You prove here that a candidate can be
refused on a measurement rather than on an opinion, that releasing is moving a
pointer and therefore reversible in one call, and that the registry can answer
"which model answered on day 23?" months later because every retrain was written
down as a run.
Where it sits: Block four — "The monitor spoke. Now what?" and "The rule that
makes the gate mean anything", and the definition slides "Definition — a run,
and the retrain that logs it", "Definition — the gate, and the margin it opens
on", "Definition — the release verdict: promote, hold or roll back",
"Definition — release by alias, and rollback", "Definition — the model
signature" and "Definition — the feedback loop".
What the check grades: retrain reproduces service.models.train on the pooled
days and logs exactly one run carrying the training window as parameters and the
gate-day accuracy as a metric; ask_champion is answered on a reordered frame and
refused on a renamed or missing column; passes_gate refuses a tie, refuses a
worse candidate, refuses everything at a margin of 0.95 and scores each model on
its own feature list; promote writes the artefact, moves the pickle registry and
moves the champion alias to the same version, and rollback moves both back;
close_the_loop ends at "v3" with a history of ['v1', 'v2', 'v3'] at the shipped
margin of 0.01, and at "v2" at a margin of 0.05; and release_verdict makes the
right call on eight situations whose answers differ and four one-quantity
changes, with a reason built out of the numbers it was handed.
Needs: pandas, mlflow.exceptions, and lab_support and service.models for the
days, the model and the two registries.

Twenty-five minutes.

The monitor has done its job. It said, on day 21, that the model was getting
things wrong while the inputs sat still. Now somebody has to fix it, and a fix
that cannot be undone is not a fix, it is a gamble.

Module 3 built the machinery already: a registry that says which version is
approved, and a gate that refuses to promote a candidate unless it measures
better than the model it would replace. Nothing about that changes here. What
changes is what the candidate is, and there are two honest answers:

    fresh rows          keep the same four inputs, train on the last few days
    the explaining      add the variable that accounts for the change — here
    variable            `crew`, which is the archive's `mode` under another name

The first is the reflex, and it works: the model gets most of its accuracy back
by learning the new relationship from new examples. The second is the repair,
because the reason the relationship changed is that a human took the wheel, and
a model that cannot see who is driving has to guess.

Measure both. Do not argue about them.

One rule about the measuring, and it is the one that gets broken: gate on days
neither model was trained on. A candidate evaluated on its own training days
always wins, and a gate that always opens is a rubber stamp.

Two registries, side by side, and this lab writes both:

    the fifty-line one  `service/artefacts/registry.json` and one pickle per
                        version — Module 3's, so the mechanism stays visible
    the platform one    a local MLflow store, `mlruns.db` and `mlartifacts/`.
                        Every training is a *run*; the released version carries
                        the alias `champion`; promotion moves the alias and
                        rollback moves it back. No server, no account.

The check compares them: the pickle registry's `approved` and the alias must
name the same model. That comparison is why the toy registry is kept — it shows
what the platform is doing underneath.

What you write: retrain, ask_champion, passes_gate, promote, rollback,
close_the_loop and release_verdict.

This lab depends on no other lab. It can be written first.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from mlflow.exceptions import MlflowException

from lab_support import NotSolved, approved_model, day_frame, load_champion  # noqa: E402
from service import models                                          # noqa: E402

LAB = 4

# The days the loop uses. Trained on the first stretch, gated on the second, and
# the second is deliberately outside the first.
TRAIN_DAYS = (21, 24)
GATE_DAYS = (25, 27)

# How much better a candidate has to be before it replaces what is in service.
# A bar of nought promotes noise; a bar this high refuses a real improvement.
# The number is a decision, and it belongs beside the result.
MARGIN = 0.01

# The three calls a release decision may make, and there is no fourth.
RELEASE_CALLS = ("promote", "hold", "roll back")


def _days(first_day: int, last_day: int) -> pd.DataFrame:
    """The pooled rows of days first_day..last_day inclusive."""
    return pd.concat([day_frame(number) for number in range(first_day, last_day + 1)],
                     ignore_index=True)


def retrain(first_day: int, last_day: int, features=None):
    """Train a candidate on days first_day..last_day inclusive, and log the run.

    Use service.models.train, which fixes the seed and the forest's shape, so
    that the only thing changing between candidates is the data and the feature
    list. `features` of None means service.models.FEATURES — the four the model
    in service uses.

    Then record it: service.models.log_run(model, features, training_frame,
    gate_frame) writes one run carrying the training window, the feature list,
    the seed and the row count as parameters, the accuracy on the training days
    and on the gate days as metrics, and the model itself with its signature and
    its pinned environment. It returns the registered version and stamps it on
    the model, which is what promote() needs a moment later. Return the model.

    Definition graded by the check:
        run = (parameters, data window, metrics, artefact, environment) recorded once and never edited; one retrain, one run
        (Zaharia et al., 2018; Pineau et al., 2021). Choices: the data window is
        the first and last day of the training frame, read off the frame rather
        than passed in, so it cannot disagree with the rows; the gate days go in
        as a metric, never as the training score. Slide: "Definition — a run, and
        the retrain that logs it".
    Needs: _days, service.models.train, service.models.log_run, service.models.FEATURES
    """
    # TODO: pool the days, train, then log the run.
    raise NotSolved("retrain(first_day, last_day, features) still raises instead of "
                    "returning a fitted model")


def ask_champion(request):
    """Ask the model the `champion` alias names. Return (answers, status).

    `request` is a frame whose columns are already the model's inputs. Load the
    released model with load_champion() from lab_support — that is
    `models:/aboard@champion`, a lookup rather than a path — and call
    `.predict(request)` on it.

    Return `(answers, "answered")` when it answers, and `(None, "refused")` when
    schema enforcement rejects the frame. Catch MlflowException, which is what
    the refusal is raised as, and nothing wider: a refusal is information, and
    swallowing every exception turns a bug into a shrug.

    Run it three times and watch what the signature is for: a frame whose
    columns are in a different order is silently corrected by name; a frame with
    a column renamed, or missing, is refused before a single row is scored.

    Definition graded by the check:
        signature = (column names, types, order) inferred at logging and enforced at predict: a reorder is corrected, a rename or an absence is refused
        (Zaharia et al., 2018). Choice: the pyfunc model, not the raw estimator
        — the raw one accepts a reordered frame silently and answers nonsense.
        Slide: "Definition — the model signature".
    Needs: load_champion, MlflowException
    """
    # TODO: load through the alias, predict, and report what happened.
    raise NotSolved("ask_champion(request) still raises instead of returning "
                    "(answers, status)")


def passes_gate(champion, champion_features, candidate, candidate_features,
                frame, margin: float = MARGIN) -> bool:
    """Would you release this candidate? Measure both on `frame` and decide.

    Return True only when the candidate's accuracy is at least `margin` above
    the champion's. Equal is not better, and slightly better is not better
    enough — that is what the margin is for.

    Use service.models.accuracy, passing each model its own feature list. The
    two lists differ whenever a candidate adds a variable, and handing a model
    the wrong columns is the training-and-serving mismatch from Module 3.

    Definition graded by the check:
        promote ⟺ accuracy(candidate) ≥ accuracy(champion) + margin, both measured on gate days neither model was trained on, each model on its own feature list
        (Kreuzberger, Kühl & Hirschl, 2023). Choices: margin = 0.01 by this
        course's decision, where Module 3's gate used nought — nought promotes a
        tie, and a tie is noise. The principled number is the Wilson half-width
        on however many labels the gate days hold, which is where a margin
        should come from when you have the count. Slide: "Definition — the gate,
        and the margin it opens on".
    Needs: service.models.accuracy
    """
    # TODO: two accuracies, one comparison.
    raise NotSolved("passes_gate(champion, ..., candidate, ..., frame, margin) still "
                    "raises instead of returning True or False")


def promote(version: str, model, features) -> dict:
    """Release a candidate: save it, point both registries at it, keep the history.

    On the pickle side: service.models.save writes the artefact, then load the
    registry, set "approved" to `version`, append `version` to "history" if it
    is not already the last entry, and save it.

    On the platform side: service.models.version_of(model) is the registered
    version retrain() stamped on this object, and service.models.set_champion
    moves the alias to it. Record it under the registry's "mlflow_version" map
    as well, so the two stores can be compared by content later.

    Return the updated registry.

    Release is moving a pointer. Nothing is copied over anything, and the model
    that was in service a minute ago is still on disk — which is what makes the
    next function possible.

    Definition graded by the check:
        release(v): alias champion ↦ v, the artefact never overwritten
        (Zaharia et al., 2018; Schelter et al., 2018). Choices: the history
        appends and never replaces; both registries move together, so the
        comparison between them means something. Slide: "Definition — release by
        alias, and rollback".
    Needs: service.models.save, service.models.load_registry, service.models.save_registry, service.models.version_of, service.models.set_champion
    """
    # TODO: save, then move both pointers.
    raise NotSolved("promote(version, model, features) still raises instead of "
                    "returning the updated registry")


def rollback() -> dict:
    """Point both registries back at the version before the approved one.

    Return the updated registry. If the approved version is already the first
    entry in the history, there is nothing to go back to: leave it alone and
    return the registry unchanged.

    Move the alias back too, with service.models.set_champion, using the
    "mlflow_version" the registry recorded for the version you are stepping back
    to. A rollback that moves one pointer and not the other leaves the two
    stores disagreeing about what is in service, which is worse than either.

    Leave the history as it is. Rolling back does not un-release anything; it
    records that you released and then thought better of it.

    Definition graded by the check:
        rollback: alias champion ↦ v_previous, the same move reversed, with the history append-only
        (Zaharia et al., 2018; Schelter et al., 2018). Choice: the previous entry
        in the history, not "the highest version below this one" — the history
        records what was actually released, in order. Slide: "Definition —
        release by alias, and rollback".
    Needs: service.models.load_registry, service.models.save_registry, service.models.set_champion
    """
    # TODO: find the approved version in the history, step back one, move both.
    raise NotSolved("rollback() still raises instead of returning the registry")


def close_the_loop(margin: float = MARGIN) -> dict:
    """The whole cycle, once. Return a record of what happened.

    1. Take the model in service as the champion, from approved_model().
    2. Retrain on TRAIN_DAYS with the same features. Gate it on GATE_DAYS.
       If it passes, promote it as "v2" — it is now the champion.
    3. Retrain on TRAIN_DAYS again, this time with service.models.
       EXPLAINED_FEATURES, which adds `crew`. Gate that against whatever is
       now the champion, on the same days. If it passes, promote it as "v3".
    4. Return a dict with these keys:

           champion        the accuracy of the model that started in service
           fresh_rows      the accuracy of the first candidate
           with_crew       the accuracy of the second
           approved        the version the registry ends up pointing at

    All three accuracies are measured on GATE_DAYS, which nothing was trained on.

    Step 3 says "whatever is now the champion" and means it. Gate the second
    candidate against the first, not against the model you started the morning
    with — otherwise the second release is justified by a comparison against
    something already retired. Run it again with margin=0.05 and watch what
    changes: the second candidate is better than the first by 2.6 points, which
    clears a one-point bar and does not clear a five-point one.

    Definition graded by the check:
        the model's own decisions enter the data it is next trained on: D_{t+1} = f(D_t, model_t), so what "normal" means is partly the previous model's doing
        (Perdomo, Zrnic, Mendler-Dünner & Hardt, 2020; Gama et al., 2014). The
        honest limit, stated: `service/world.py` does not shape its labels by
        what the model predicted, so the loop's structure here is real and its
        mechanism is not simulated. On a real shuttle it would be — a passenger
        the model says there is no room for is a passenger who does not board,
        and that row is in tomorrow's training set. Slide: "Definition — the
        feedback loop".
    Needs: approved_model, retrain, passes_gate, promote, service.models.accuracy, service.models.EXPLAINED_FEATURES
    """
    # TODO: two candidates, two gates, one pointer.
    raise NotSolved("close_the_loop() still raises instead of returning a record")


def release_verdict(evidence: dict) -> tuple:
    """Return (call, reason). `call` is one of "promote", "hold", "roll back".

    passes_gate answers one arithmetic question: is this candidate better by
    enough? A release decision is a larger question, and the extra parts of it
    are what the gate cannot see. `evidence` holds exactly these keys:

        champion_accuracy          what is in service, on the gate days
        candidate_accuracy         the candidate, on the same days
        margin                     the bar this course chose, MARGIN above
        wilson_half_width          the half-width of the interval the labels
                                   behind those accuracies support
        gate_labels                how many rows those accuracies were measured on
        measured_on_training_days  True when the candidate's number came from
                                   days it was trained on
        champion_regressed         True when what is in service is measurably
                                   worse than the version it replaced

    Four questions, in this order, and the order is the content:

    The reason has to *name* the quantities it weighed, so write them out —
    "candidate accuracy 0.8831 against champion accuracy 0.6881, on 3600 gate
    labels" names three of them and quotes only numbers you were handed.

        1. Is what is *already released* worse than what it replaced? Then stop
           reading about candidates. Call `roll back` — it is a pointer move and
           it takes seconds, and production is losing accuracy while you think.
        2. Did the candidate's number come from days it was trained on? Then
           there is no admissible measurement here at all. Call `hold` and
           measure again on days neither model saw.
        3. Is the gain at least the margin *and* at least the Wilson half-width
           the labels support? A gain smaller than the half-width is a gain the
           measurement cannot distinguish from nought, whatever the stated bar
           says. Call `promote`.
        4. Otherwise call `hold`.

    Question three is the slide's sentence made into code: the principled margin
    is the Wilson half-width on however many labels the gate days hold, and the
    stated margin is a floor under it rather than a substitute for it.

    The reason is graded as hard as the call: at least forty characters, every
    number in it one from `evidence`, and at least two of the quantities you
    weighed named. Do not modify `evidence`.

    Definition graded by the check:
        verdict = roll back if the released model is worse than the one it replaced · hold if the candidate's number came from days it was trained on · promote if candidate − champion ≥ max(margin, Wilson half-width) · hold otherwise
        (Kreuzberger, Kühl & Hirschl, 2023; Wilson, 1927). Choices: rollback is
        asked first, because a released regression costs accuracy while a
        candidate is being argued about; the half-width is a floor under the
        margin rather than a replacement for it, so a stated bar can be stricter
        than the measurement but never looser. Slide: "Definition — the release
        verdict: promote, hold or roll back".
    Needs: the seven keys above, and nothing else
    """
    # TODO: four questions in order, then a reason built from `evidence`.
    raise NotSolved("release_verdict(evidence) still raises instead of returning "
                    "(call, reason)")


if __name__ == "__main__":
    import json

    before = json.dumps(models.load_registry())
    record = close_the_loop()
    print(f"trained on days {TRAIN_DAYS[0]}-{TRAIN_DAYS[1]}, "
          f"gated on days {GATE_DAYS[0]}-{GATE_DAYS[1]}, margin {MARGIN}\n")
    print(f"  the model in service           {record['champion']:.4f}")
    print(f"  retrained on fresh rows        {record['fresh_rows']:.4f}")
    print(f"  plus the explaining variable   {record['with_crew']:.4f}")
    print(f"\n  approved after the loop        {record['approved']}")
    print(f"  the champion alias names       version {models.champion_version()}")
    print(f"  registry before                {before}")
    print(f"  after a rollback               {json.dumps(rollback())}")
    print(f"  the alias now names            version {models.champion_version()}")
