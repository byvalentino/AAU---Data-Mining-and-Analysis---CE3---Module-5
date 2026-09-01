#!/usr/bin/env python3
"""Generate phone traces with the archive's shape and none of its people.

    python3 data/make_phones.py

The real `passengers.csv` is the position trace of sixteen identifiable
volunteers. Under Article 4 of the General Data Protection Regulation that is
personal data, and it is not in this repository and will not be.

What is here instead is a generator whose every parameter was measured from the
real file by `Module 2/slides/measure.py` and written into `calibration.json`.
The magnitudes ship; the rows never do. A generator with invented parameters
would be a fabricated experimental result, so nothing below is invented — where
a number appears it came from the archive, and `calibration.json` says so.

What is faithfully reproduced, because the labs turn on it:

  * two sampling rates -- phones about once a second, vehicles twice a second,
    which is the alignment problem in Lab 1
  * beacons absent at the measured rates, and absent *for a reason*: out of
    range, not lost in transit. That is a mechanism, not noise (Lab 2)
  * the same absence encoded twice -- an empty signal strength and a proximity
    of -1 mark exactly the same rows, as they do in the archive
  * a `bus_id` present exactly when the passenger is aboard, which is the leak
    Lab 3 hunts. In the archive this is real: 7,723 rows aboard all carry it,
    6,000 rows not aboard all lack it, with no exceptions
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CALIBRATION = json.loads((HERE / "calibration.json").read_text())

SEED = 20200122
BEACONS = ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]
PROXIMITY = {"rssiA": "proxA", "rssiB": "proxB", "rssiC": "proxC",
             "rssi1": "prox1", "rssi2": "prox2"}

# A beacon is heard when it is near. Signal strength falls with distance, so the
# absence is a consequence of geometry rather than of a dropped packet -- which
# is exactly what makes it missing-not-at-random.
RANGE_METRES = 25.0
STRENGTH_AT_ONE_METRE = -45.0
PATH_LOSS = 2.4


def generate(day: str = "2020-01-22", seed: int = SEED,
             with_truth: bool = False) -> pd.DataFrame:
    """One day of phone traces. Set with_truth to keep the hidden columns."""
    rng = np.random.default_rng(seed)
    phones = CALIBRATION["phones_per_day"][day]
    interval = CALIBRATION["phone_interval_s"]
    # The labelled rows are the basis for everything below, because that is the
    # population the per-state beacon shares were measured over. The all-rows
    # absence share in calibration.json is a different population -- it includes
    # the second day, which carries no labels at all.
    aboard_share = CALIBRATION["aboard_share_of_labelled"] / 100

    start = pd.Timestamp(f"{day} 09:00:00", tz="UTC")
    per_phone = 900  # fifteen minutes each, enough for windows without being slow

    frames = []
    for phone in range(phones):
        clock = start + pd.to_timedelta(np.arange(per_phone) * interval, unit="s")
        # Nanosecond resolution, matching what pd.to_datetime gives on the
        # vehicle file. Mismatched resolutions make merge_asof refuse, which is
        # a real nuisance but not the lesson of any lab here.
        clock = clock.astype("datetime64[ns, UTC]")

        # A journey: waiting at a stop, riding, then waiting again. The rider is
        # near the vehicle beacons only while aboard.
        # One ride per phone, long enough that the aboard share matches the
        # archive's labelled rows: 56.3 per cent aboard, 43.7 not.
        ride = int(round(aboard_share * per_phone))
        boarding = int(rng.integers(0, max(1, per_phone - ride)))
        position = np.arange(per_phone)
        is_aboard = (position >= boarding) & (position < boarding + ride)

        # Distance to each beacon. Note what this does NOT do: tie hearing a
        # vehicle beacon to being aboard. The archive says that link is not
        # there -- "this beacon was heard" agrees with "aboard" on 42 to 48 per
        # cent of labelled rows, against a base rate of 56.3. The whole trial
        # fits in a box about 67 by 86 metres (Module 1), and beacon range is
        # tens of metres, so being near a stop beacon and being on the vehicle
        # are not separable by proximity here.
        #
        # So closeness is its own process, and each beacon is heard at the rate
        # the archive measured for each state. What survives, because it is real,
        # is that the readings which ARE absent are the far and weak ones.
        distances = {}
        for name in BEACONS:
            wander = rng.normal(0, 1.0, per_phone).cumsum()
            wander = (wander - wander.min()) / (np.ptp(wander) or 1.0)
            distances[name] = 1.0 + wander * 95.0

        frame = pd.DataFrame({
            "phone_id": f"p{phone:02d}",
            "timestamp_utc": clock,
            # The same trap as the archive: a column named `timestamp` that is
            # local time, sitting beside the one that is not.
            "timestamp": clock.tz_convert("Europe/Copenhagen").tz_localize(None),
            "speed": np.where(is_aboard, rng.uniform(0, 3.5, per_phone),
                              rng.uniform(0, 1.4, per_phone)).round(3),
            "stationary": (~is_aboard).astype(int),
        })

        for name in BEACONS:
            distance = np.clip(distances[name], 0.5, None)
            true_strength = (STRENGTH_AT_ONE_METRE
                             - 10 * PATH_LOSS * np.log10(distance)
                             + rng.normal(0, 2.0, per_phone))

            # Heard at the archive's measured rate, separately for aboard and
            # not-aboard rows, and within each state the strongest are heard.
            # Both the marginal absence rate and the (weak) association with the
            # label then match the real file rather than a story about it.
            heard = np.zeros(per_phone, dtype=bool)
            for state, share_table in ((True, CALIBRATION["beacon_heard_when_aboard"]),
                                       (False, CALIBRATION["beacon_heard_when_not_aboard"])):
                rows = is_aboard if state else ~is_aboard
                count = int(rows.sum())
                if count == 0:
                    continue
                keep = int(round(share_table[name] / 100 * count))
                if keep == 0:
                    continue
                cutoff = np.sort(true_strength[rows])[::-1][keep - 1]
                heard |= rows & (true_strength >= cutoff)

            frame[f"{name}_true"] = true_strength.round(1)
            frame[name] = np.where(heard, true_strength.round(1), np.nan)
            # The same absence, encoded a second way: -1 rather than empty.
            bands = np.select(
                [true_strength > -60, true_strength > -75], [1, 2], default=3)
            frame[PROXIMITY[name]] = np.where(heard, bands, -1)

        frame["label"] = np.where(is_aboard, "Bus 1", "Stop C")
        frame["label2"] = np.where(is_aboard, "IN", "OUT")
        # The leak. Present exactly when aboard, as in the archive.
        frame["bus_id"] = np.where(is_aboard, "VJRD1A10224000055", None)
        frame["aboard_truth"] = is_aboard
        frames.append(frame)

    everything = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp_utc", "phone_id"]).reset_index(drop=True)

    if with_truth:
        return everything
    hidden = [c for c in everything.columns if c.endswith("_true")] + ["aboard_truth"]
    return everything.drop(columns=hidden)


if __name__ == "__main__":
    phones = generate()
    print(f"{len(phones):,} rows, {phones.shape[1]} columns, "
          f"{phones['phone_id'].nunique()} phones")
    print("\nabsent share, generated against the archive's measured share:")
    for name in BEACONS:
        measured = CALIBRATION["beacon_absent_share"][name]
        print(f"  {name:7} generated {phones[name].isna().mean() * 100:5.1f} %"
              f"   archive {measured:5.1f} %")
    same = all((phones[r].isna() == (phones[p] == -1)).all() for r, p in PROXIMITY.items())
    print(f"\nabsence encoded twice, and the two agree everywhere: {same}")
