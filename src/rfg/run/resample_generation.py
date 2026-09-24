"""Shared resumability and deterministic-seeding helpers for resampling jobs."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def parse_seeds(value: str, n: int) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if n < 1:
        raise ValueError("n must be positive")
    if len(seeds) < n:
        raise ValueError(f"need {n} seeds, received {len(seeds)}")
    return seeds[:n]


def indexed_seeds(value: str, n: int, candidate_offset: int = 0) -> list[tuple[int, int]]:
    """Pair fixed seeds with contiguous candidate indices for incremental pools."""
    if candidate_offset < 0:
        raise ValueError("candidate_offset must be non-negative")
    return list(enumerate(parse_seeds(value, n), start=candidate_offset))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_complete(json_path: Path, wav_path: Path) -> bool:
    try:
        record = json.loads(json_path.read_text())
        return not record.get("error") and wav_path.stat().st_size > 44
    except (OSError, json.JSONDecodeError):
        return False


def atomic_json(path: Path, record: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    os.replace(temporary, path)
