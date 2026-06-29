"""SOH prediction pipeline with the original GRU-DSConv model structure.

The core model remains:
GRU -> SinusoidalEncoding -> DepthwiseSeparableConv1d
-> MultiScaleDilatedStack -> MemoryAugmentedModule -> FC.

The main changes are intentionally around data flow:
- remove discharge capacity from model inputs to avoid SOH label leakage
- use [voltage, current, dV] while keeping NUM_VAR = 3
- build battery/cycle metadata for timeline evaluation and anomaly detection
- normalize validation/test with train-set statistics only
- use Huber loss and squeeze(-1) in training
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # Allows preprocessing checks where torch is absent.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ModuleNotFoundError:
    IsolationForest = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None


FIXED_LEN = 300
EARLY_CYCLE = 20
HORIZON = 0
NUM_VAR = 3
FEATURE_MODE = "practical"
FEATURE_NAMES_BY_MODE = {
    "practical": ["voltage", "current", "dV"],
    "oracle_capacity": ["voltage", "current", "capacity"],
}
FEATURE_NAMES = FEATURE_NAMES_BY_MODE[FEATURE_MODE]
REFERENCE_CYCLE_COUNT = 5
DEFAULT_SEED = 42


def set_feature_mode(mode: str) -> None:
    """Switch feature construction without changing model channel count."""
    global FEATURE_MODE, FEATURE_NAMES
    if mode not in FEATURE_NAMES_BY_MODE:
        raise ValueError(f"unknown feature mode: {mode}")
    FEATURE_MODE = mode
    FEATURE_NAMES = FEATURE_NAMES_BY_MODE[mode]


@dataclass(frozen=True)
class SplitData:
    X: np.ndarray
    y: np.ndarray
    meta: list[dict]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resample_cycle(features: np.ndarray, fixed_len: int = FIXED_LEN) -> np.ndarray:
    """Linearly resample one cycle from (T, C) to (fixed_len, C)."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"expected 2D cycle features, got shape={features.shape}")
    if features.shape[0] < 2:
        raise ValueError("not enough points to resample")

    old_x = np.linspace(0.0, 1.0, features.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, fixed_len, dtype=np.float32)
    cols = [np.interp(new_x, old_x, features[:, i]) for i in range(features.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def _finite_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr[np.isfinite(arr)]


def extract_features(cycle: dict, fixed_len: int = FIXED_LEN) -> np.ndarray:
    """Build cycle features.

    practical: [V, I, dV], no discharge capacity leakage.
    oracle_capacity: [V, I, Q], intentionally includes capacity for leakage ablation.
    """
    voltage = _finite_1d(cycle.get("voltage_in_V", []))
    current = _finite_1d(cycle.get("current_in_A", []))

    if FEATURE_MODE == "oracle_capacity":
        capacity = _finite_1d(cycle.get("discharge_capacity_in_Ah", []))
        min_len = min(len(voltage), len(current), len(capacity))
    else:
        capacity = None
        min_len = min(len(voltage), len(current))

    if min_len < 5:
        raise ValueError("not enough feature points")

    voltage = voltage[:min_len]
    current = current[:min_len]

    if FEATURE_MODE == "oracle_capacity":
        features = np.stack([voltage, current, capacity[:min_len]], axis=1)
    else:
        d_voltage = np.gradient(voltage).astype(np.float32)
        features = np.stack([voltage, current, d_voltage], axis=1)
    return resample_cycle(features, fixed_len=fixed_len)


def extract_soh_label(cycle: dict, reference_capacity: float) -> float:
    """SOH = current cycle discharge capacity / reference capacity."""
    q = _finite_1d(cycle.get("discharge_capacity_in_Ah", []))
    if len(q) < 5:
        raise ValueError("not enough discharge capacity points")

    soh = float(np.nanmax(q) / reference_capacity)
    if not (0.0 < soh < 1.5):
        raise ValueError(f"abnormal SOH label: {soh:.3f}")
    return soh


def infer_reference_capacity(cell: dict, cycles: list[dict]) -> float:
    """Use nominal capacity when present; otherwise fall back to early-cycle Qd."""
    nominal = cell.get("nominal_capacity_in_Ah")
    if nominal is not None and np.isfinite(float(nominal)) and float(nominal) > 0:
        return float(nominal)

    early_q = []
    for cycle in cycles[:REFERENCE_CYCLE_COUNT]:
        q = _finite_1d(cycle.get("discharge_capacity_in_Ah", []))
        if len(q) >= 5:
            early_q.append(float(np.nanmax(q)))
    if not early_q:
        raise ValueError("could not infer reference capacity")
    return float(np.mean(early_q))


def load_cell_file(path: str | Path, fixed_len: int = FIXED_LEN) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load one BatteryLife pkl file and return cycle features, SOH, metadata."""
    path = Path(path)
    with path.open("rb") as f:
        cell = pickle.load(f)

    cycles = list(cell.get("cycle_data", []))
    cycles = sorted(cycles, key=lambda c: int(c.get("cycle_number", len(cycles))))
    reference_capacity = infer_reference_capacity(cell, cycles)
    cell_id = str(cell.get("cell_id", path.stem))

    X_cycles, soh_values, meta = [], [], []
    for idx, cycle in enumerate(cycles):
        try:
            features = extract_features(cycle, fixed_len=fixed_len)
            soh = extract_soh_label(cycle, reference_capacity)
            X_cycles.append(features)
            soh_values.append(soh)
            meta.append(
                {
                    "file": path.name,
                    "cell_id": cell_id,
                    "cycle_number": int(cycle.get("cycle_number", idx + 1)),
                    "reference_capacity": reference_capacity,
                }
            )
        except ValueError:
            continue

    if len(X_cycles) == 0:
        raise ValueError(f"no usable cycles in {path}")
    return np.stack(X_cycles), np.asarray(soh_values, dtype=np.float32), meta


def build_dataset(
    files: Iterable[str | Path],
    early_cycle: int = EARLY_CYCLE,
    horizon: int = HORIZON,
    fixed_len: int = FIXED_LEN,
) -> SplitData:
    """Build sliding-window samples while preserving battery/cycle metadata."""
    X_list, y_list, meta_list = [], [], []

    for fp in files:
        try:
            cycles, soh, cycle_meta = load_cell_file(fp, fixed_len=fixed_len)
        except Exception as exc:
            print(f"[skip] {fp}: {exc}")
            continue

        n = len(cycles)
        if n < early_cycle + horizon:
            print(f"[skip] {fp}: only {n} usable cycles")
            continue

        for end in range(early_cycle - 1, n - horizon):
            target_idx = end + horizon
            X_list.append(cycles[end - early_cycle + 1 : end + 1])
            y_list.append(soh[target_idx])

            item = dict(cycle_meta[target_idx])
            item.update(
                {
                    "input_start_cycle": cycle_meta[end - early_cycle + 1]["cycle_number"],
                    "input_end_cycle": cycle_meta[end]["cycle_number"],
                    "target_cycle": cycle_meta[target_idx]["cycle_number"],
                    "horizon": horizon,
                }
            )
            meta_list.append(item)

    if not X_list:
        raise ValueError("no samples built")

    return SplitData(
        X=np.stack(X_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        meta=meta_list,
    )


def subset_split_data(split: SplitData, indices: Sequence[int], split_name: str) -> SplitData:
    idx = np.asarray(indices, dtype=np.int64)
    meta = []
    for i in idx:
        item = dict(split.meta[int(i)])
        item["sample_split"] = split_name
        meta.append(item)
    return SplitData(
        X=split.X[idx].astype(np.float32),
        y=split.y[idx].astype(np.float32),
        meta=meta,
    )


def concatenate_split_data(parts: Sequence[SplitData]) -> SplitData:
    if not parts:
        raise ValueError("no split data parts to concatenate")
    return SplitData(
        X=np.concatenate([part.X for part in parts], axis=0).astype(np.float32),
        y=np.concatenate([part.y for part in parts], axis=0).astype(np.float32),
        meta=[item for part in parts for item in part.meta],
    )


def chronological_sample_counts(n: int, val_ratio: float = 0.15, test_ratio: float = 0.15) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError("need at least 3 samples for train/val/test split")
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = 1, 1, n - 2
    return n_train, n_val, n_test


def chronological_gap_sample_counts(
    n: int,
    gap_samples: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[int, int, int]:
    """Compute split sizes after reserving train/val and val/test gaps."""
    if gap_samples < 0:
        raise ValueError("gap_samples must be non-negative")
    usable = n - (2 * gap_samples)
    if usable < 3:
        raise ValueError(
            f"need at least {3 + 2 * gap_samples} samples for train/gap/val/gap/test split; got {n}"
        )
    return chronological_sample_counts(usable, val_ratio=val_ratio, test_ratio=test_ratio)


def build_chronological_splits_within_files(
    files: Iterable[str | Path],
    early_cycle: int = EARLY_CYCLE,
    horizon: int = HORIZON,
    fixed_len: int = FIXED_LEN,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[SplitData, SplitData, SplitData, list[dict]]:
    """Build train/val/test splits inside every pkl file in cycle order."""
    train_parts, val_parts, test_parts = [], [], []
    split_details = []

    for fp in files:
        try:
            one = build_dataset([fp], early_cycle=early_cycle, horizon=horizon, fixed_len=fixed_len)
        except Exception as exc:
            print(f"[skip split] {fp}: {exc}")
            continue

        n = len(one.y)
        if n < 3:
            print(f"[skip split] {fp}: only {n} samples")
            continue

        order = sorted(
            range(n),
            key=lambda i: (
                int(one.meta[i].get("target_cycle", 0)),
                int(one.meta[i].get("input_end_cycle", 0)),
            ),
        )
        n_train, n_val, n_test = chronological_sample_counts(n, val_ratio=val_ratio, test_ratio=test_ratio)
        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]

        train_part = subset_split_data(one, train_idx, "train")
        val_part = subset_split_data(one, val_idx, "val")
        test_part = subset_split_data(one, test_idx, "test")
        train_parts.append(train_part)
        val_parts.append(val_part)
        test_parts.append(test_part)

        def target_range(part: SplitData) -> list[int | None]:
            values = [int(item.get("target_cycle", 0)) for item in part.meta]
            return [min(values), max(values)] if values else [None, None]

        fp_path = Path(fp)
        split_details.append(
            {
                "file": fp_path.name,
                "domain": infer_battery_domain(fp_path),
                "n_samples": n,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
                "train_target_cycle_range": target_range(train_part),
                "val_target_cycle_range": target_range(val_part),
                "test_target_cycle_range": target_range(test_part),
            }
        )

    return (
        concatenate_split_data(train_parts),
        concatenate_split_data(val_parts),
        concatenate_split_data(test_parts),
        split_details,
    )


def infer_experiment_condition(path: str | Path) -> str:
    """Infer a condition-level label from a raw sample filename.

    This keeps the dataset/protocol information and strips likely cell or
    replicate identifiers such as trailing letters, battery numbers, or cell IDs.
    The label is intentionally conservative and is saved in split_info.json for QA.
    """
    stem = Path(path).stem
    rules = [
        (r"^(CALB_.+)-\d+$", r"\1"),
        (r"^(CALCE_[A-Za-z0-9-]+)_\d+$", r"\1"),
        (r"^(HNEI_.+)_[a-z]$", r"\1"),
        (r"^(MICH_.+)_[a-z]$", r"\1"),
        (r"^(SNL_.+)_[a-z]$", r"\1"),
        (r"^(UL-PUR_.+)_[a-z]$", r"\1"),
        (r"^(HUST)_\d+-\d+$", r"\1"),
        (r"^(ISU-ILCC)_[A-Za-z0-9-]+$", r"\1"),
        (r"^(MATR)_b\d+c\d+$", r"\1"),
        (r"^(NA-ion_.+)_\d+_\d+$", r"\1"),
        (r"^(RWTH)_\d+$", r"\1"),
        (r"^(SDU_Battery)_\d+$", r"\1"),
        (r"^(Stanford_Nova_Regular(?:_Ref)?)_\d+$", r"\1"),
        (r"^(Tongji\d+_[^_]+)_.+$", r"\1"),
        (r"^(XJTU_[^_]+)_battery-\d+$", r"\1"),
        (r"^(ZN-coin_.+)_\d+_Batch-\d+$", r"\1"),
    ]
    for pattern, repl in rules:
        if re.match(pattern, stem):
            return re.sub(pattern, repl, stem)

    parts = stem.split("_")
    if len(parts) > 1 and re.fullmatch(r"[a-z]|\d{1,4}|battery-\d+", parts[-1]):
        return "_".join(parts[:-1])
    return stem


def group_files_by_experiment_condition(files: Sequence[str | Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        fp = Path(path)
        groups.setdefault(infer_experiment_condition(fp), []).append(fp)
    return groups


def split_files_by_experiment_condition(
    files: Sequence[str | Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Condition-group split: all files from the same condition stay together."""
    groups = group_files_by_experiment_condition(files)
    condition_names = sorted(groups)
    if len(condition_names) < 3:
        counts = ", ".join(f"{name}={len(groups[name])}" for name in condition_names)
        raise ValueError(
            "need at least 3 experiment-condition groups for train/val/test split. "
            f"Available groups: {counts}"
        )

    rng = random.Random(seed)
    rng.shuffle(condition_names)
    n = len(condition_names)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = 1, 1, n - 2

    train_conditions = condition_names[:n_train]
    val_conditions = condition_names[n_train : n_train + n_val]
    test_conditions = condition_names[n_train + n_val :]

    def flatten(selected: Sequence[str]) -> list[Path]:
        paths = [path for condition in selected for path in groups[condition]]
        return sorted(paths)

    return flatten(train_conditions), flatten(val_conditions), flatten(test_conditions)


def build_condition_gap_splits_within_files(
    files: Iterable[str | Path],
    early_cycle: int = EARLY_CYCLE,
    horizon: int = HORIZON,
    fixed_len: int = FIXED_LEN,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    gap_samples: int = 5,
) -> tuple[SplitData, SplitData, SplitData, list[dict]]:
    """Build condition-aware chronological splits with unused gap windows.

    Each preprocessed pkl file is first tagged with a coarse dataset label and a
    finer experiment-condition label. Samples are then split inside that file in
    target-cycle order as train | gap | val | gap | test. Groups/files that do
    not have enough samples are reported in split_details and excluded.
    """
    paths = [Path(p) for p in files]
    train_parts, val_parts, test_parts = [], [], []
    split_details = []

    def target_range_from_meta(items: Sequence[dict]) -> list[int | None]:
        values = [int(item.get("target_cycle", 0)) for item in items]
        return [min(values), max(values)] if values else [None, None]

    for fp in paths:
        domain = infer_battery_domain(fp)
        condition = infer_experiment_condition(fp)
        detail = {
            "file": fp.name,
            "battery_domain": domain,
            "experiment_condition": condition,
            "split_gap_samples": gap_samples,
            "required_min_samples": 3 + 2 * gap_samples,
        }
        try:
            one = build_dataset([fp], early_cycle=early_cycle, horizon=horizon, fixed_len=fixed_len)
        except Exception as exc:
            detail.update({"status": "skipped", "reason": str(exc), "total_samples": 0})
            split_details.append(detail)
            print(f"[skip condition-gap split] {fp}: {exc}")
            continue

        n = len(one.y)
        detail["total_samples"] = n
        try:
            n_train, n_val, n_test = chronological_gap_sample_counts(
                n,
                gap_samples=gap_samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        except Exception as exc:
            detail.update({"status": "skipped", "reason": str(exc)})
            split_details.append(detail)
            print(f"[skip condition-gap split] {fp}: {exc}")
            continue

        order = sorted(
            range(n),
            key=lambda i: (
                int(one.meta[i].get("target_cycle", 0)),
                int(one.meta[i].get("input_end_cycle", 0)),
            ),
        )
        train_end = n_train
        val_start = train_end + gap_samples
        val_end = val_start + n_val
        test_start = val_end + gap_samples

        train_idx = order[:train_end]
        train_val_gap_idx = order[train_end:val_start]
        val_idx = order[val_start:val_end]
        val_test_gap_idx = order[val_end:test_start]
        test_idx = order[test_start:]

        train_part = subset_split_data(one, train_idx, "train")
        val_part = subset_split_data(one, val_idx, "val")
        test_part = subset_split_data(one, test_idx, "test")
        for part in (train_part, val_part, test_part):
            for item in part.meta:
                item["battery_domain"] = domain
                item["experiment_condition"] = condition
                item["split_gap_samples"] = gap_samples

        train_parts.append(train_part)
        val_parts.append(val_part)
        test_parts.append(test_part)

        def target_range(indices: Sequence[int]) -> list[int | None]:
            return target_range_from_meta([one.meta[int(i)] for i in indices])

        detail.update(
            {
                "status": "used",
                "train_samples": len(train_idx),
                "train_val_gap_samples": len(train_val_gap_idx),
                "val_samples": len(val_idx),
                "val_test_gap_samples": len(val_test_gap_idx),
                "test_samples": len(test_idx),
                "train_target_cycle_range": target_range(train_idx),
                "train_val_gap_target_cycle_range": target_range(train_val_gap_idx),
                "val_target_cycle_range": target_range(val_idx),
                "val_test_gap_target_cycle_range": target_range(val_test_gap_idx),
                "test_target_cycle_range": target_range(test_idx),
            }
        )
        split_details.append(detail)

    if not train_parts or not val_parts or not test_parts:
        skipped = [item for item in split_details if item.get("status") == "skipped"]
        reasons = "; ".join(f"{item['file']}: {item.get('reason', 'unknown')}" for item in skipped[:5])
        raise ValueError(
            "condition-gap split produced no usable train/val/test samples. "
            f"First skipped reasons: {reasons}"
        )

    return (
        concatenate_split_data(train_parts),
        concatenate_split_data(val_parts),
        concatenate_split_data(test_parts),
        split_details,
    )


def split_files(
    files: Sequence[str | Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Battery-level split: a pkl file belongs to exactly one split."""
    paths = [Path(p) for p in files]
    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)
    if n < 3:
        raise ValueError("need at least 3 battery files for train/val/test split")

    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = 1, 1, n - 2

    train = paths[:n_train]
    val = paths[n_train : n_train + n_val]
    test = paths[n_train + n_val :]
    return train, val, test


def infer_battery_domain(path: str | Path) -> str:
    """Infer a coarse dataset/domain label from a raw sample filename."""
    stem = Path(path).stem
    prefix = stem.split("_", 1)[0]
    if prefix.startswith("Tongji"):
        return "Tongji"
    return prefix


def group_files_by_domain(files: Sequence[str | Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        fp = Path(path)
        groups.setdefault(infer_battery_domain(fp), []).append(fp)
    return groups


def split_files_same_domain_eval(
    files: Sequence[str | Path],
    seed: int = DEFAULT_SEED,
    eval_domain: str | None = None,
) -> tuple[list[Path], list[Path], list[Path], str]:
    """Hold out one domain, then split that same domain into val/test files."""
    paths = [Path(p) for p in files]
    if len(paths) < 3:
        raise ValueError("need at least 3 battery files for train/val/test split")

    groups = group_files_by_domain(paths)
    eligible_domains = sorted(domain for domain, items in groups.items() if len(items) >= 2)
    if not eligible_domains:
        counts = ", ".join(f"{domain}={len(items)}" for domain, items in sorted(groups.items()))
        raise ValueError(
            "same-domain eval split needs at least one inferred domain with 2+ pkl files. "
            f"Available domain counts: {counts}"
        )

    if eval_domain:
        domain_by_lower = {domain.lower(): domain for domain in groups}
        selected_domain = domain_by_lower.get(eval_domain.lower())
        if selected_domain is None:
            raise ValueError(
                f"unknown eval domain: {eval_domain}. "
                f"Available domains: {', '.join(sorted(groups))}"
            )
        if len(groups[selected_domain]) < 2:
            raise ValueError(
                f"eval domain {selected_domain!r} has only {len(groups[selected_domain])} file(s); "
                "need at least 2 for validation/test."
            )
    else:
        max_count = max(len(groups[domain]) for domain in eligible_domains)
        largest_domains = [domain for domain in eligible_domains if len(groups[domain]) == max_count]
        rng = random.Random(seed)
        rng.shuffle(largest_domains)
        selected_domain = largest_domains[0]

    rng = random.Random(seed)
    eval_files = list(groups[selected_domain])
    rng.shuffle(eval_files)
    n_val = max(1, len(eval_files) // 2)
    val = eval_files[:n_val]
    test = eval_files[n_val:]
    if not test:
        raise ValueError(f"eval domain {selected_domain!r} did not leave any test files")

    train = [fp for domain, items in groups.items() if domain != selected_domain for fp in items]
    if not train:
        raise ValueError(f"eval domain {selected_domain!r} consumes all files; no train files remain")
    rng.shuffle(train)
    return train, val, test, selected_domain


def normalize_by_train(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Channel-wise normalization with train statistics only."""
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


class SOHDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise ModuleNotFoundError("torch is required to create SOHDataset")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


if nn is not None:

    class SinusoidalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 2048):
            super().__init__()
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]


    class DepthwiseSeparableConv1d(nn.Module):
        def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
            super().__init__()
            padding = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=channels,
                ),
                nn.Conv1d(channels, channels, kernel_size=1),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class MultiScaleDilatedStack(nn.Module):
        def __init__(self, channels: int, dilations: Sequence[int] = (1, 2, 4), dropout: float = 0.1):
            super().__init__()
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                            groups=channels,
                        ),
                        nn.Conv1d(channels, channels, kernel_size=1),
                        nn.BatchNorm1d(channels),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for dilation in dilations
                ]
            )
            self.mix = nn.Conv1d(channels * len(dilations), channels, kernel_size=1)
            self.norm = nn.BatchNorm1d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.cat([branch(x) for branch in self.branches], dim=1)
            return self.norm(self.mix(out) + x)


    class MemoryAugmentedModule(nn.Module):
        def __init__(self, channels: int, memory_slots: int = 16):
            super().__init__()
            self.memory = nn.Parameter(torch.randn(memory_slots, channels) * 0.02)
            self.query = nn.Linear(channels, channels)
            self.out = nn.Sequential(nn.Linear(channels * 2, channels), nn.GELU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pooled = x.mean(dim=1)
            query = self.query(pooled)
            attn = torch.softmax(query @ self.memory.t() / math.sqrt(x.size(-1)), dim=-1)
            memory_context = attn @ self.memory
            return self.out(torch.cat([pooled, memory_context], dim=-1))


    class GRUDSConvSOH(nn.Module):
        """Original model skeleton preserved: GRU -> DSConv -> Dilated -> Memory -> FC."""

        def __init__(
            self,
            num_var: int = NUM_VAR,
            fixed_len: int = FIXED_LEN,
            n_cycles: int = EARLY_CYCLE,
            gru_hidden: int = 64,
            channels: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.num_var = num_var
            self.fixed_len = fixed_len
            self.n_cycles = n_cycles
            self.gru = nn.GRU(
                input_size=num_var,
                hidden_size=gru_hidden,
                batch_first=True,
                bidirectional=False,
            )
            self.project = nn.Linear(gru_hidden, channels)
            self.pos = SinusoidalEncoding(channels, max_len=max(n_cycles + 32, 128))
            self.dsconv = DepthwiseSeparableConv1d(channels, dropout=dropout)
            self.dilated = MultiScaleDilatedStack(channels, dropout=dropout)
            self.map = MemoryAugmentedModule(channels)
            self.fc = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, cycles, fixed_len, num_var)
            batch, cycles, length, num_var = x.shape
            x = x.reshape(batch * cycles, length, num_var)
            _, h = self.gru(x)
            cycle_emb = self.project(h[-1]).reshape(batch, cycles, -1)
            cycle_emb = self.pos(cycle_emb)

            z = cycle_emb.transpose(1, 2)
            z = self.dsconv(z)
            z = self.dilated(z).transpose(1, 2)
            z = self.map(z)
            return self.fc(z)

else:

    class GRUDSConvSOH:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required to use GRUDSConvSOH")


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for model training/evaluation")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    require_torch()
    return DataLoader(SOHDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str | None = None,
    patience: int = 0,
    min_delta: float = 0.0,
    huber_delta: float = 0.02,
    clip_grad_norm: float = 1.0,
    lr_scheduler_patience: int = 0,
    lr_scheduler_factor: float = 0.5,
):
    require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
        )

    history = []
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b).squeeze(-1)
            loss = criterion(pred, y_b)
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "lr": current_lr,
        }
        history.append(row)
        print(f"epoch={epoch:03d} train={row['train_loss']:.5f} val={val_loss:.5f} lr={current_lr:.3g}")

        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"early_stop epoch={epoch:03d} best_val={best_val:.5f} "
                f"patience={patience} min_delta={min_delta:g}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_loss(model, loader, criterion, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(X_b).squeeze(-1)
            losses.append(float(criterion(pred, y_b).detach().cpu()))
    return float(np.mean(losses))


def predict_loader(model, loader, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            out = model(X_b.to(device)).squeeze(-1).cpu().numpy()
            preds.append(out)
            targets.append(y_b.numpy())

    return np.concatenate(preds), np.concatenate(targets)


def make_result_df(meta: list[dict], y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(meta).copy()
    df["actual_soh"] = y_true
    df["pred_soh"] = y_pred
    df["residual"] = np.abs(df["actual_soh"] - df["pred_soh"])
    return df


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def acc_15(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) <= 0.15))


def eol_cycle_error(df: pd.DataFrame, threshold: float = 0.8) -> float:
    errors = []
    for _, g in df.groupby("cell_id"):
        g = g.sort_values("target_cycle")
        actual_cross = g[g["actual_soh"] <= threshold]
        pred_cross = g[g["pred_soh"] <= threshold]
        if len(actual_cross) > 0 and len(pred_cross) > 0:
            actual_eol = int(actual_cross.iloc[0]["target_cycle"])
            pred_eol = int(pred_cross.iloc[0]["target_cycle"])
            errors.append(abs(actual_eol - pred_eol))
    return float("nan") if not errors else float(np.mean(errors))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, result_df: pd.DataFrame | None = None) -> dict:
    if mean_absolute_error is None:
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = float("nan")
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_percent": mape(y_true, y_pred) * 100.0,
        "15pct_Acc": acc_15(y_true, y_pred),
    }
    if result_df is not None:
        metrics["EOL_Error_cycles"] = eol_cycle_error(result_df)
    return metrics


def add_slope_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["cell_id", "target_cycle"]).copy()
    out["actual_slope"] = out.groupby("cell_id")["actual_soh"].diff().fillna(0.0)
    out["pred_slope"] = out.groupby("cell_id")["pred_soh"].diff().fillna(0.0)
    out["slope_score"] = np.abs(out["actual_slope"] - out["pred_slope"])
    return out


def robust_z(score: np.ndarray, reference: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    med = np.median(reference)
    mad = np.median(np.abs(reference - med)) + 1e-8
    return (score - med) / (1.4826 * mad)


def extract_embeddings(model, loader, device: str | None = None) -> np.ndarray:
    """Extract MemoryAugmentedModule output with a hook, without changing forward()."""
    require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []

    def hook_fn(_module, _inputs, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = model.map.register_forward_hook(hook_fn)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for X_b, _ in loader:
            _ = model(X_b.to(device))
    handle.remove()

    return np.concatenate(embeddings, axis=0)


def add_anomaly_scores(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_latent_score: np.ndarray | None = None,
    test_latent_score: np.ndarray | None = None,
    threshold_quantile: float = 0.975,
) -> tuple[pd.DataFrame, float]:
    """Combine residual, slope, and optional latent scores using robust z-scores."""
    val_df = add_slope_score(val_df)
    test_df = add_slope_score(test_df)

    val_df["residual_z"] = robust_z(val_df["residual"].values, val_df["residual"].values).clip(min=0)
    test_df["residual_z"] = robust_z(test_df["residual"].values, val_df["residual"].values).clip(min=0)
    val_df["slope_z"] = robust_z(val_df["slope_score"].values, val_df["slope_score"].values).clip(min=0)
    test_df["slope_z"] = robust_z(test_df["slope_score"].values, val_df["slope_score"].values).clip(min=0)

    score_cols = ["residual_z", "slope_z"]
    if val_latent_score is not None and test_latent_score is not None:
        val_df["latent_z"] = robust_z(val_latent_score, val_latent_score).clip(min=0)
        test_df["latent_z"] = robust_z(test_latent_score, val_latent_score).clip(min=0)
        score_cols.append("latent_z")

    val_anomaly = val_df[score_cols].mean(axis=1)
    threshold = float(np.quantile(val_anomaly, threshold_quantile))
    test_df["anomaly_score"] = test_df[score_cols].mean(axis=1)
    test_df["is_anomaly"] = test_df["anomaly_score"] > threshold
    return test_df, threshold


def fit_latent_iforest(
    model,
    val_loader,
    test_loader,
    device: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if IsolationForest is None:
        print("[warn] sklearn is unavailable; skipping latent IsolationForest score")
        return None, None

    val_emb = extract_embeddings(model, val_loader, device=device)
    test_emb = extract_embeddings(model, test_loader, device=device)
    detector = IsolationForest(n_estimators=200, contamination="auto", random_state=DEFAULT_SEED)
    detector.fit(val_emb)
    return -detector.score_samples(val_emb), -detector.score_samples(test_emb)


def plot_cell_timeline(result_df: pd.DataFrame, cell_id: str | None = None) -> None:
    import matplotlib.pyplot as plt

    if cell_id is None:
        cell_id = str(result_df["cell_id"].value_counts().index[0])

    g = result_df[result_df["cell_id"] == cell_id].sort_values("target_cycle")
    plt.figure(figsize=(10, 5))
    plt.plot(g["target_cycle"], g["actual_soh"], label="Actual SOH")
    plt.plot(g["target_cycle"], g["pred_soh"], "--", label="Predicted SOH")
    plt.axhline(0.8, linestyle=":", label="EOL SOH=0.8")
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.title(f"SOH timeline: {cell_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_pkl_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.rglob("*.pkl"))


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    files = find_pkl_files(args.data_dir)
    if args.max_files:
        files = files[: args.max_files]
    print(f"found {len(files)} pkl files")

    train_files, val_files, test_files = split_files(files, seed=args.seed)
    print(f"split files: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    train = build_dataset(train_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
    val = build_dataset(val_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
    test = build_dataset(test_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)

    X_train, X_val, X_test, X_mean, X_std = normalize_by_train(train.X, val.X, test.X)
    print(f"sample shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"feature names: {FEATURE_NAMES}")
    print(f"train normalization mean={X_mean.reshape(-1)} std={X_std.reshape(-1)}")

    train_loader = make_loader(X_train, train.y, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, val.y, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, test.y, batch_size=args.batch_size, shuffle=False)

    model = GRUDSConvSOH(
        num_var=NUM_VAR,
        fixed_len=args.fixed_len,
        n_cycles=args.early_cycle,
    )
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
    )

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    val_pred, val_true = predict_loader(model, val_loader, device=device)
    test_pred, test_true = predict_loader(model, test_loader, device=device)
    val_df = make_result_df(val.meta, val_true, val_pred)
    test_df = make_result_df(test.meta, test_true, test_pred)

    metrics = regression_metrics(test_true, test_pred, test_df)
    print(pd.Series(metrics).to_string())

    val_latent, test_latent = fit_latent_iforest(model, val_loader, test_loader, device=device)
    anomaly_df, threshold = add_anomaly_scores(val_df, test_df, val_latent, test_latent)
    print(f"anomaly threshold={threshold:.4f}, flagged={int(anomaly_df['is_anomaly'].sum())}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / "train_history.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    anomaly_df.to_csv(out_dir / "test_anomaly_scores.csv", index=False)
    np.save(out_dir / "x_mean.npy", X_mean)
    np.save(out_dir / "x_std.npy", X_std)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--fixed-len", type=int, default=FIXED_LEN)
    parser.add_argument("--early-cycle", type=int, default=EARLY_CYCLE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
