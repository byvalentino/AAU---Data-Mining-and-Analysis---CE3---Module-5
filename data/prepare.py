#!/usr/bin/env python3
"""Prepare every dataset and artefact the labs read, and write data/MANIFEST.json.

    python3 data/prepare.py          called by setup.sh; safe to re-run

Generated data is generated here, once, deterministically (seed 20200122), and
written to data/ as Parquet, so that (a) a lab never depends on a generator
running at import time, (b) `make check` can verify the data is present before
grading anything (verify/check_00_data.py), and (c) a teacher can open the file
a student worked on. Shipped data (the archive slice) is verified, not rewritten.

The manifest records, per file, the row count, the column count and a hash of
the values -- not of the bytes, because Parquet metadata carries a writer string.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent          # exercises/data
EXERCISES = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXERCISES))

MANIFEST = HERE / "MANIFEST.json"
SLICE = HERE / "bus_slice.csv.gz"


def content_hash(frame: pd.DataFrame) -> str:
    return f"{int(pd.util.hash_pandas_object(frame, index=False).sum()):x}"


def table_entry(path: pathlib.Path, frame: pd.DataFrame, source: str, note: str) -> dict:
    return {"path": str(path.relative_to(EXERCISES)), "kind": "table", "source": source,
            "rows": int(len(frame)), "columns": int(frame.shape[1]),
            "content_hash": content_hash(frame), "note": note}


def main() -> int:
    files = []

    # 1. The archive slice ships with every module; verify it, never rewrite it.
    if not SLICE.exists():
        print(f"prepare failed: {SLICE.relative_to(EXERCISES)} is missing from the checkout")
        return 1
    bus = pd.read_csv(SLICE, low_memory=False)
    files.append(table_entry(SLICE, bus, "archive",
                             "vehicle VJRD1A10224000055, 22-23 January 2020, as shipped"))
    print(f"verified  {SLICE.name}: {len(bus)} rows x {bus.shape[1]} columns")

    # 2. Generated phone traces, where the module has the generator.
    if (HERE / "make_phones.py").exists():
        from make_phones import generate, CALIBRATION
        for day in CALIBRATION["phones_per_day"]:
            frame = generate(day=day, with_truth=False)
            out = HERE / f"phones_{day}.parquet"
            frame.to_parquet(out, index=False)
            files.append(table_entry(out, frame, "generated",
                                     f"make_phones.generate(day={day!r}), seed 20200122"))
            print(f"generated {out.name}: {len(frame)} rows x {frame.shape[1]} columns")

    # 3. The generated stream, where the module has a world.
    if (EXERCISES / "service" / "world.py").exists():
        from service import world
        frame = world.stream()
        out = HERE / "stream.parquet"
        frame.to_parquet(out, index=False)
        files.append(table_entry(out, frame, "generated",
                                 "service.world.stream(), 28 days, seed 20200122"))
        print(f"generated {out.name}: {len(frame)} rows x {frame.shape[1]} columns")

    # 4. Trained artefacts, where the module serves a model.
    if (EXERCISES / "service" / "models.py").exists():
        result = subprocess.run([sys.executable, str(EXERCISES / "service" / "models.py")],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("prepare failed: service/models.py did not train --")
            print(result.stdout[-1500:], result.stderr[-1500:])
            return 1
        artefacts = EXERCISES / "service" / "artefacts"
        for path in sorted(artefacts.glob("*")):
            if path.is_file():
                files.append({"path": str(path.relative_to(EXERCISES)), "kind": "artefact",
                              "source": "generated", "bytes": path.stat().st_size,
                              "note": "written by service/models.py"})
        print(f"trained   {len([f for f in files if f['kind'] == 'artefact'])} artefact(s) "
              f"in service/artefacts/")

        # The platform registry beside the fifty-line one. Recorded as artefacts
        # rather than as tables: run identifiers and timestamps inside the store
        # are random, so there is no content hash to compare -- what setup
        # promised is that the store exists, holds a logged model, and is the one
        # the labs read. Module 3 records the same two entries the same way.
        store, artefact_root = EXERCISES / "mlruns.db", EXERCISES / "mlartifacts"
        if not store.exists() or not any(artefact_root.rglob("MLmodel")):
            print("prepare failed: service/models.py did not write the MLflow store "
                  "(mlruns.db and mlartifacts/)")
            return 1
        files.append({"path": "mlruns.db", "kind": "artefact", "source": "generated",
                      "bytes": store.stat().st_size,
                      "note": "MLflow tracking and registry store, sqlite, written by "
                              "service/models.py -- one run for v1, registered model "
                              "'aboard', alias champion on version 1"})
        files.append({"path": "mlartifacts", "kind": "directory", "source": "generated",
                      "files": sum(1 for p in artefact_root.rglob("*") if p.is_file()),
                      "note": "MLflow artefacts: the registered model with its signature, "
                              "input example and pinned environment"})
        print("recorded  mlruns.db and mlartifacts/ (the MLflow store)")

    MANIFEST.write_text(json.dumps({"seed": 20200122, "files": files}, indent=1))
    print(f"wrote     data/MANIFEST.json -- {len(files)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
