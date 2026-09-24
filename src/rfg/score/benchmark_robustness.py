"""Duration-stratified paired robustness statistics for speech benchmarks."""
from __future__ import annotations

import random
import statistics
from collections.abc import Callable


def duration_group(duration: float) -> str:
    if duration <= 15:
        return "le15"
    if duration <= 30:
        return "15to30"
    return "gt30"


def bootstrap_mean_ci(values: list[float], *, seed: int, n_boot: int) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    draws = sorted(
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(n_boot)
    )
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]]


def summarize_rows(
    rows: list[dict], *, seed: int = 20260920, n_boot: int = 10_000,
) -> dict:
    """Summarize internal and ASR correctness with paired accuracy drops."""
    if not rows:
        raise ValueError("cannot summarize an empty row set")
    sources = ("internal", "asr1", "asr2", "asr3")
    result = {
        "n": len(rows),
        "duration_sec": {
            "mean": statistics.mean(row["duration_sec"] for row in rows),
            "median": statistics.median(row["duration_sec"] for row in rows),
            "max": max(row["duration_sec"] for row in rows),
        },
        "accuracy": {},
        "paired_drop_from_internal": {},
    }
    for source in sources:
        result["accuracy"][source] = statistics.mean(
            float(row["correct"][source]) for row in rows
        )
    for index, source in enumerate(sources[1:], 1):
        differences = [
            float(row["correct"]["internal"]) - float(row["correct"][source])
            for row in rows
        ]
        result["paired_drop_from_internal"][source] = {
            "mean": statistics.mean(differences),
            "ci95": bootstrap_mean_ci(
                differences, seed=seed + index, n_boot=n_boot,
            ),
        }
    return result


def stratified_summary(
    rows: list[dict], *, seed: int = 20260920, n_boot: int = 10_000,
) -> dict:
    selectors: dict[str, Callable[[dict], bool]] = {
        "all": lambda row: True,
        "le15": lambda row: duration_group(row["duration_sec"]) == "le15",
        "15to30": lambda row: duration_group(row["duration_sec"]) == "15to30",
        "le30": lambda row: row["duration_sec"] <= 30,
        "gt30": lambda row: duration_group(row["duration_sec"]) == "gt30",
    }
    output = {}
    for index, (name, selector) in enumerate(selectors.items()):
        selected = [row for row in rows if selector(row)]
        if selected:
            output[name] = summarize_rows(
                selected, seed=seed + index * 10, n_boot=n_boot,
            )
        else:
            output[name] = {"n": 0}
    return output
