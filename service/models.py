#!/usr/bin/env python3
"""The model the monitor watches, and the two registries it is released through.

    python3 service/models.py

Module 3 built a registry and a gate. Module 5 uses them, unchanged in shape:
release is moving a pointer, and a candidate is promoted only if it measures
better than the model it would replace.

Two registries live side by side here, on purpose:

    the fifty-line one   service/artefacts/registry.json and one pickle per
                         version -- Module 3's, so the mechanism stays visible
    the platform one     a local MLflow store: mlruns.db (SQLite, tracking and
                         registry) and mlartifacts/ (the logged models). Every
                         training is a run; the released version carries the
                         alias `champion`; promotion moves the alias, rollback
                         moves it back. Nothing is a server and nothing needs
                         an account -- both files sit beside the exercises and
                         are gitignored.

Lab 4 writes to both and check 4 compares them: the pickle registry's `approved`
and the alias must name the same version. That comparison is the point of
keeping the toy one -- it shows what the platform is doing underneath.

What is new in what the candidate is: block four retrains, and there are two
ways to do it:

    fresh rows          keep the same four features and train on recent data
    the explaining      add the variable that explains the change -- here
    variable            `crew`, which is the archive's `mode` under another name

Retraining on fresh rows is the reflex. Adding the variable is the repair. The
lab measures both rather than arguing about them.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import pathlib
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = pathlib.Path(__file__).resolve().parent
EXERCISES = HERE.parent
ARTEFACTS = HERE / "artefacts"

# The MLflow store. Overridden by use_store() when a demonstration wants a
# throw-away copy rather than the module's own.
MLRUNS_DB = EXERCISES / "mlruns.db"
MLARTIFACTS = EXERCISES / "mlartifacts"
EXPERIMENT = "aboard"      # every run goes into this experiment
MODEL_NAME = "aboard"      # the registered model whose numbered versions are the releases
CHAMPION = "champion"      # the alias that names the version in service

# Left at 7 rather than aligned to the course seed: every accuracy on the slides,
# in the checks and in the notebook was measured with it, and re-measuring all
# of them buys nothing a student can see.
SEED = 7
FEATURES = ["speed", "rssi1", "rssi2", "rssiC"]
EXPLAINED_FEATURES = FEATURES + ["crew"]
TARGET = "aboard"

# An absent beacon reading is filled with a value no real reading takes, so the
# model can learn "not heard" as its own state rather than as a middling signal.
ABSENT = -100.0


def prepare(frame: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    # A frame, not an array: the column names travel with the values, so the
    # model's signature can be inferred from them when a run is logged and
    # enforced by name when the registered model answers a request.
    #
    # Every column as a double, including the crew flag, which is 0 or 1. An
    # integer column in a signature cannot carry a missing value, so a request
    # with one hole in it would be refused for the wrong reason; declaring the
    # inputs as doubles says what the model actually consumes.
    return frame[list(features)].fillna(ABSENT).astype("float64")


def train(frame: pd.DataFrame, features=FEATURES) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                   random_state=SEED, n_jobs=1)
    model.fit(prepare(frame, features), frame[TARGET])
    return model


def accuracy(model, frame: pd.DataFrame, features=FEATURES) -> float:
    return float((model.predict(prepare(frame, features)) == frame[TARGET]).mean())


# ---- the MLflow store: tracking, the registered model, the alias -------------

def use_store(directory) -> None:
    """Point this module at another store -- a temporary one for a demonstration
    that must not touch the module's own runs."""
    global MLRUNS_DB, MLARTIFACTS
    directory = pathlib.Path(directory)
    MLRUNS_DB, MLARTIFACTS = directory / "mlruns.db", directory / "mlartifacts"


def tracking():
    """Point MLflow at the module's local store and return (mlflow, MlflowClient()).

    The same function as Module 3's `open_store()`, and deliberately so: one
    store layout, one way of quieting it, across both modules.

    Imported lazily, because mlflow costs most of a second to import and checks
    1 and 3 never need it. Two quiet-downs, both deliberate. MLflow resets its
    own logger to INFO when it is imported, so the level is set after the import
    and not before. And the first touch of a new store runs a schema migration
    that narrates itself on stderr through a logging configuration file which
    also installs a handler on the root logger; that narration is not this
    module's story, so it goes to a sink and the handler it left behind is
    removed.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    for noisy in ("mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    uri = "sqlite:///" + str(MLRUNS_DB)
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    with contextlib.redirect_stderr(io.StringIO()):
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            MLARTIFACTS.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(EXPERIMENT, artifact_location=MLARTIFACTS.as_uri())
    for handler in root_logger.handlers:
        if handler not in handlers_before:
            root_logger.removeHandler(handler)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow, MlflowClient()


def environment_pins() -> list:
    """The environment a run records: the versions actually installed here.
    Pinned explicitly rather than inferred by MLflow, which spawns a
    subprocess per model and costs three seconds a run."""
    import cloudpickle
    import mlflow
    import sklearn
    return [f"scikit-learn=={sklearn.__version__}", f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}", f"cloudpickle=={cloudpickle.__version__}",
            f"mlflow=={mlflow.__version__}"]


def log_run(model, features, training: pd.DataFrame, gate: pd.DataFrame = None) -> str:
    """Record one training as a run and register the model as the next version.

    The run carries the parameters (the training window read off the frame's
    `day` column, the feature list, the seed, the row count), the metrics (the
    accuracy on the training days and, when a gate frame is given, on the gate
    days), and the model with its signature -- column names and types inferred
    from the prepared frame -- plus the pinned environment. Written once; never
    edited (Zaharia et al., 2018; Pineau et al., 2021).

    Returns the registered version, e.g. "2", and stamps it on the model as
    `model.mlflow_version` (with `model.mlflow_run_id`), so that promote() can
    move the alias to the version this very object was registered as.
    """
    from mlflow.models import infer_signature

    mlflow, _ = tracking()
    features = list(features)
    inputs = prepare(training, features)
    first_day, last_day = int(training["day"].min()), int(training["day"].max())
    parameters = {"first_day": first_day, "last_day": last_day,
                  "features": ",".join(features), "seed": SEED,
                  "rows": int(len(training)),
                  "n_estimators": model.n_estimators, "max_depth": model.max_depth}
    metrics = {"accuracy_training_days": accuracy(model, training, features)}
    if gate is not None:
        parameters["gate_first_day"] = int(gate["day"].min())
        parameters["gate_last_day"] = int(gate["day"].max())
        metrics["accuracy_gate_days"] = accuracy(model, gate, features)

    # Registering a version prints two lines straight to stderr rather than
    # through logging, so the level set in tracking() cannot reach them; they go
    # to a sink here, exactly as Module 3 does it.
    name = f"days {first_day}-{last_day}, {len(features)} features"
    with contextlib.redirect_stderr(io.StringIO()), mlflow.start_run(run_name=name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(inputs, model.predict(inputs)),
            registered_model_name=MODEL_NAME, input_example=inputs.head(2),
            pip_requirements=environment_pins())
    version = str(info.registered_model_version)
    model.mlflow_run_id, model.mlflow_version = run.info.run_id, version
    return version


def version_of(model) -> str:
    """The registered version a model was logged as -- the stamp log_run left."""
    version = getattr(model, "mlflow_version", None)
    if version is None:
        raise ValueError(
            "this model was never recorded as a run, so it has no registered "
            "version to release: train candidates with retrain(), which calls "
            "service.models.log_run, before promoting them")
    return str(version)


def champion_version():
    """The version the `champion` alias names, or None when nothing is released."""
    from mlflow.exceptions import MlflowException

    _, client = tracking()
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except MlflowException:
        return None


def set_champion(version) -> None:
    """Release by alias: point `champion` at a version. Rollback is the same call
    with the previous version. Left alone when it already points there, so a
    rebuild that changes nothing writes nothing."""
    version = str(version)
    if champion_version() == version:
        return
    _, client = tracking()
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)


def load_champion():
    """The released model, loaded through the alias -- the model that answers now.
    A pyfunc model: it enforces the signature, reordering columns by name and
    refusing a request whose columns are renamed or missing."""
    mlflow, _ = tracking()
    with contextlib.redirect_stderr(io.StringIO()):
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")


def lineage() -> pd.DataFrame:
    """One row per registered version: what it was trained on, how it scored, and
    whether it is the champion. The registry as a table, read back from the store.

    The alias comes from the registered model rather than from each version.
    `search_model_versions` returns versions whose `aliases` list is empty even
    when an alias points at them (verified, mlflow 2.17.2 on the SQLite store),
    so reading it there would report no champion at all.
    """
    _, client = tracking()
    aliases = client.get_registered_model(MODEL_NAME).aliases or {}
    champion = str(aliases.get(CHAMPION, ""))
    rows = []
    for entry in client.search_model_versions(f"name = '{MODEL_NAME}'"):
        run = client.get_run(entry.run_id)
        rows.append({
            "version": int(entry.version),
            "run_id": entry.run_id[:8],
            "first_day": int(run.data.params.get("first_day", -1)),
            "last_day": int(run.data.params.get("last_day", -1)),
            "features": run.data.params.get("features", ""),
            "accuracy_training_days": run.data.metrics.get("accuracy_training_days", np.nan),
            "accuracy_gate_days": run.data.metrics.get("accuracy_gate_days", np.nan),
            "champion": str(entry.version) == champion,
        })
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


def run_count() -> int:
    """How many runs the experiment holds. Checks count these before and after a
    retrain: one retrain, one run."""
    mlflow, _ = tracking()
    return int(len(mlflow.search_runs(experiment_names=[EXPERIMENT])))


def _reference_run():
    """The run that already records v1 -- days 0-9, the four features, the seed --
    as (run_id, version), or None. Found by content, never by run id."""
    mlflow, client = tracking()
    if not MLRUNS_DB.exists():
        return None
    matches = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=(f"params.first_day = '0' and params.last_day = '9' and "
                       f"params.features = '{','.join(FEATURES)}' and "
                       f"params.seed = '{SEED}'"),
        order_by=["attributes.start_time ASC"])
    for run_id in matches["run_id"] if len(matches) else []:
        versions = client.search_model_versions(f"run_id = '{run_id}'")
        if versions:
            return run_id, str(min(int(v.version) for v in versions))
    return None


# ---- snapshot and restore, for anything that writes and must leave no trace ---

def snapshot_stores() -> pathlib.Path:
    """Copy service/artefacts/, mlruns.db and mlartifacts/ aside. Returns the
    directory; hand it to restore_stores() in a `finally`."""
    where = pathlib.Path(tempfile.mkdtemp(prefix="m5_stores_"))
    if ARTEFACTS.exists():
        shutil.copytree(ARTEFACTS, where / "artefacts")
    if MLRUNS_DB.exists():
        shutil.copy2(MLRUNS_DB, where / "mlruns.db")
    if MLARTIFACTS.exists():
        shutil.copytree(MLARTIFACTS, where / "mlartifacts")
    return where


def restore_stores(where) -> None:
    """Put back exactly what snapshot_stores() saved: files added since are
    removed, files changed since are overwritten, and a store that did not
    exist is removed again."""
    where = pathlib.Path(where)
    for saved, live in ((where / "artefacts", ARTEFACTS),
                        (where / "mlartifacts", MLARTIFACTS)):
        if live.exists():
            shutil.rmtree(live)
        if saved.exists():
            shutil.copytree(saved, live)
    saved = where / "mlruns.db"
    if saved.exists():
        # Copied over the same path rather than unlinked and re-created: MLflow
        # keeps its SQLite connection open for the life of the process, and a
        # connection to an unlinked file goes on writing into the void.
        shutil.copyfile(saved, MLRUNS_DB)
    elif MLRUNS_DB.exists():
        MLRUNS_DB.unlink()
    shutil.rmtree(where, ignore_errors=True)


# ---- the fifty-line registry ----------------------------------------------

def build() -> dict:
    """Train v1 on the reference period and register it as approved -- in both
    registries.

    Safe to call again. The pickle side is rewritten to v1 only. On the MLflow
    side the run that already records v1 is found by its content and reused
    rather than logged twice, and the champion alias is pointed back at it;
    runs logged since stay, because a run is written once and never edited.
    """
    from service.world import reference_period

    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    reference = reference_period(10)
    model = train(reference)

    found = _reference_run()
    if found is None:
        version = log_run(model, FEATURES, reference)
    else:
        # The same data, seed and estimator give the same forest, so the run
        # already in the store is this model's run.
        model.mlflow_run_id, version = found
        model.mlflow_version = version
    set_champion(version)

    path = ARTEFACTS / "model_v1.pkl"
    path.write_bytes(pickle.dumps({"model": model, "features": list(FEATURES)}))

    record = {"v1": {"version": "v1", "features": list(FEATURES),
                     "trained_on_days": "0-9",
                     "accuracy": round(accuracy(model, reference), 4),
                     "rows": int(len(reference)), "seed": SEED,
                     "mlflow_version": version, "run_id": model.mlflow_run_id}}
    (ARTEFACTS / "metrics.json").write_text(json.dumps(record, indent=1))
    (ARTEFACTS / "registry.json").write_text(json.dumps(
        {"approved": "v1", "history": ["v1"], "mlflow_version": {"v1": version}},
        indent=1))
    return record


def load(version: str = "v1") -> dict:
    return pickle.loads((ARTEFACTS / f"model_{version}.pkl").read_bytes())


def save(version: str, model, features) -> None:
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    (ARTEFACTS / f"model_{version}.pkl").write_bytes(
        pickle.dumps({"model": model, "features": list(features)}))


def load_registry() -> dict:
    return json.loads((ARTEFACTS / "registry.json").read_text())


def save_registry(registry: dict) -> None:
    (ARTEFACTS / "registry.json").write_text(json.dumps(registry, indent=1))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent))
    for version, facts in build().items():
        print(f"{version}  trained on days {facts['trained_on_days']}, "
              f"{facts['rows']:,} rows, accuracy {facts['accuracy']:.4f}, "
              f"registered as {MODEL_NAME} version {facts['mlflow_version']} "
              f"(alias {CHAMPION}), run {facts['run_id'][:8]}")
