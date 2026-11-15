"""Run and aggregate the approved communication-frequency experiment."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

from .spade_simulation import (
    SimulationConfig,
    build_budget_schedule,
    mean_ci95,
    run_spade,
)


RAW_FIELDS = (
    "seed",
    "schedule",
    "T",
    "tau",
    "rounds",
    "local_steps",
    "eta",
    "mu",
    "beta",
    "a0",
    "stationarity",
    "consensus",
    "criterion",
    "query_count",
    "transmitted_scalars",
)


def _schedule_label(total_local_steps, tau):
    if tau == 1:
        return "tau=1"
    rounded_root = round(total_local_steps ** (1.0 / 3.0))
    if rounded_root**3 == total_local_steps and tau == rounded_root:
        return "tau=T^(1/3)"
    return "tau={}".format(tau)


def aggregate_records(records):
    groups = {}
    for record in records:
        key = (int(record["T"]), int(record["tau"]))
        groups.setdefault(key, []).append(record)

    summary = []
    for (total_local_steps, tau), group in sorted(groups.items()):
        row = {
            "schedule": _schedule_label(total_local_steps, tau),
            "T": total_local_steps,
            "tau": tau,
            "rounds": int(group[0]["rounds"]),
            "n_seeds": len(group),
            "query_count": int(group[0]["query_count"]),
            "transmitted_scalars": int(group[0]["transmitted_scalars"]),
            "communication_reduction_vs_tau1": float(total_local_steps / group[0]["rounds"]),
        }
        for metric in ("criterion", "stationarity", "consensus"):
            mean, low, high = mean_ci95([float(item[metric]) for item in group])
            row[metric + "_mean"] = mean
            row[metric + "_ci95_low"] = low
            row[metric + "_ci95_high"] = high
        summary.append(row)
    return summary


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run_task(task):
    seed, total_local_steps, tau, cache_path = task
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), True
    result = run_spade(seed, total_local_steps, tau, SimulationConfig())
    result["schedule"] = _schedule_label(total_local_steps, tau)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(cache_path)
    return result, False


def run_all(output_dir, n_seeds=30, workers=6):
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    tasks = []
    for budget in build_budget_schedule():
        for tau in (1, budget["tau_local"]):
            for seed in range(n_seeds):
                cache_name = "seed_{:02d}_T_{}_tau_{}.json".format(seed, budget["T"], tau)
                tasks.append((seed, budget["T"], tau, cache_dir / cache_name))

    started = time.perf_counter()
    records = []
    completed = 0
    cached = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            result, from_cache = future.result()
            records.append(result)
            completed += 1
            cached += int(from_cache)
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    "completed {}/{} runs ({} cached), elapsed {:.1f}s".format(
                        completed, len(tasks), cached, elapsed
                    ),
                    flush=True,
                )

    records.sort(key=lambda row: (int(row["T"]), int(row["tau"]), int(row["seed"])))
    flat_records = [{field: row[field] for field in RAW_FIELDS} for row in records]
    summary = aggregate_records(flat_records)
    write_csv(output_dir / "communication_cost_per_seed.csv", flat_records)
    write_csv(output_dir / "communication_cost_summary.csv", summary)
    metadata = {
        "experiment": "Non-private SPADE-DFL communication-frequency diagnostic",
        "n_seeds": n_seeds,
        "seed_definition": "Independent generated datasets; tau schedules paired by integer seed",
        "budgets": build_budget_schedule(),
        "config": SimulationConfig().__dict__,
        "query_noise": 0.0,
        "release_noise": 0.0,
        "clipping_error": 0.0,
        "interval": "Arithmetic mean with two-sided 95% Student-t confidence interval",
        "hypothesis_tests": "None",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return flat_records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run_all(args.output_dir, n_seeds=args.n_seeds, workers=args.workers)


if __name__ == "__main__":
    main()
