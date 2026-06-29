from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pandas is required to read metrics CSV files. Install pandas in your Python environment "
        "or run this with the Codex bundled Python runtime."
    ) from exc


LIION_DATASETS = {
    "CALB",
    "CALCE",
    "HNEI",
    "HUST",
    "ISU_ILCC",
    "MATR",
    "MICH",
    "MICH_EXP",
    "RWTH",
    "SDU",
    "SNL",
    "Stanford",
    "Stanford_2",
    "Tongji",
    "UL_PUR",
    "XJTU",
}

EXCLUDED_DATASETS = {"NA-ion", "ZN-coin"}
DEFAULT_EXCLUDE_MODELS = "persistence"
TEST_SORT_COLUMNS = ["RMSE", "MAE"]
TUNED_OUTPUT_MARKERS = [
    "tuned",
    "optuna",
    "tuning_outputs",
    "optimization_outputs",
    "refine",
    "adaptive",
    "successive",
    "locked_test",
    "final_hybrid",
    "final_delta_hybrid",
]

DATASET_PREFIXES = [
    ("ISU-ILCC", "ISU_ILCC"),
    ("UL-PUR", "UL_PUR"),
    ("NA-ion", "NA-ion"),
    ("ZN-coin", "ZN-coin"),
    ("Stanford_Nova_Regular_Ref", "Stanford"),
    ("Stanford_", "Stanford_2"),
    ("MICH_MCForm", "MICH"),
    ("MICH_", "MICH_EXP"),
    ("Tongji", "Tongji"),
    ("CALB", "CALB"),
    ("CALCE", "CALCE"),
    ("HNEI", "HNEI"),
    ("HUST", "HUST"),
    ("MATR", "MATR"),
    ("RWTH", "RWTH"),
    ("SDU", "SDU"),
    ("SNL", "SNL"),
    ("XJTU", "XJTU"),
]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_or_inf(value: Any) -> float:
    number = finite_or_none(value)
    return number if number is not None else float("inf")


def metric_or_neg_inf(value: Any) -> float:
    number = finite_or_none(value)
    return number if number is not None else float("-inf")


def load_manifest(search_root: Path) -> dict[str, str]:
    manifest_path = search_root / "battery_dataset_examples" / "manifest.json"
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {}
    mapping = {}
    for item in raw:
        if isinstance(item, dict) and item.get("file") and item.get("dataset"):
            mapping[str(item["file"])] = str(item["dataset"])
    return mapping


def infer_dataset(filename: str, manifest: dict[str, str]) -> str:
    name = Path(filename).name
    if name in manifest:
        return manifest[name]
    for prefix, dataset in DATASET_PREFIXES:
        if name.startswith(prefix):
            return dataset
    return "unknown"


def collect_split_files(split_info: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split_name in ["train", "val", "test"]:
        key = f"{split_name}_files"
        raw = split_info.get(key, [])
        if isinstance(raw, list):
            result[split_name] = [str(item) for item in raw]
        else:
            result[split_name] = []
    return result


def load_split_info(metrics_path: Path) -> dict[str, Any]:
    split_path = metrics_path.parent / "split_info.json"
    if not split_path.exists():
        return {}
    return json.loads(split_path.read_text(encoding="utf-8"))


def audit_raw_samples(data_dir: Path, manifest: dict[str, str]) -> dict[str, Any]:
    files = sorted(data_dir.glob("*.pkl")) if data_dir.exists() else []
    rows = []
    datasets = set()
    for path in files:
        dataset = infer_dataset(path.name, manifest)
        datasets.add(dataset)
        rows.append(
            {
                "file": path.name,
                "dataset": dataset,
                "is_liion_allowed": dataset in LIION_DATASETS,
                "is_excluded_by_plan": dataset in EXCLUDED_DATASETS,
            }
        )
    present_liion = sorted(dataset for dataset in datasets if dataset in LIION_DATASETS)
    present_excluded = sorted(dataset for dataset in datasets if dataset in EXCLUDED_DATASETS)
    missing_liion = sorted(LIION_DATASETS - set(present_liion))
    return {
        "data_dir": str(data_dir),
        "n_raw_pkl_files": len(files),
        "present_liion_datasets": present_liion,
        "present_excluded_datasets": present_excluded,
        "missing_liion_datasets": missing_liion,
        "files": rows,
    }


def audit_split(
    metrics_path: Path,
    split_info: dict[str, Any],
    manifest: dict[str, str],
    data_dir: Path,
    dataset_policy: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []
    split_files = collect_split_files(split_info)
    all_files = sorted({name for names in split_files.values() for name in names})
    dataset_by_file = {name: infer_dataset(name, manifest) for name in all_files}
    datasets = sorted(set(dataset_by_file.values()))
    excluded = sorted(dataset for dataset in datasets if dataset in EXCLUDED_DATASETS)
    unknown = sorted(dataset for dataset in datasets if dataset == "unknown")
    missing_files = sorted(name for name in all_files if not (data_dir / name).exists())

    if not split_info:
        failures.append("missing split_info.json")
    if dataset_policy == "liion_only" and excluded:
        failures.append("split uses non-Li-ion datasets excluded by the plan: " + ", ".join(excluded))
    if dataset_policy == "liion_only":
        non_liion = sorted(dataset for dataset in datasets if dataset not in LIION_DATASETS)
        if non_liion:
            failures.append("split contains datasets outside the Li-ion target set: " + ", ".join(non_liion))
    if unknown:
        warnings.append("some files could not be mapped to a dataset")
    if missing_files:
        warnings.append("some split files were not found under data-dir")

    train_set = set(split_files["train"])
    val_set = set(split_files["val"])
    test_set = set(split_files["test"])
    split_mode = str(split_info.get("split_mode", "file-level" if split_info else "unknown"))
    train_val_overlap = sorted(train_set & val_set)
    train_test_overlap = sorted(train_set & test_set)
    val_test_overlap = sorted(val_set & test_set)
    repeated_files = sorted(set(train_val_overlap + train_test_overlap + val_test_overlap))

    if repeated_files:
        if split_mode == "condition-gap-within-file":
            split_gap = int(split_info.get("split_gap", 0) or 0)
            early_cycle = int(split_info.get("early_cycle", 0) or 0)
            if split_gap < 20:
                failures.append(f"condition-gap split uses split_gap={split_gap}; the plan requires gap_cycles=20")
            if early_cycle > 20:
                failures.append(f"condition-gap split uses early_cycle={early_cycle}; the plan expects early_cycle<=20")
        else:
            failures.append("same files appear in multiple splits outside condition-gap mode")

    if split_mode in {"chronological-within-file"}:
        failures.append("chronological-within-file is not an allowed final split policy in the dataset plan")
    if split_mode == "battery" or split_mode == "file-level":
        if repeated_files:
            failures.append("file-level split has overlapping train/val/test files")
    if split_mode == "condition-group":
        train_conditions = set(split_info.get("train_experiment_conditions", []))
        val_conditions = set(split_info.get("val_experiment_conditions", []))
        test_conditions = set(split_info.get("test_experiment_conditions", []))
        overlaps = {
            "train_val": sorted(train_conditions & val_conditions),
            "train_test": sorted(train_conditions & test_conditions),
            "val_test": sorted(val_conditions & test_conditions),
        }
        overlapping_conditions = sorted({item for values in overlaps.values() for item in values})
        if overlapping_conditions:
            failures.append(
                "condition-group split has overlapping experiment conditions: "
                + ", ".join(overlapping_conditions)
            )
    if not test_set:
        failures.append("split_info has no test_files")
    if "file_experiment_conditions" not in split_info:
        warnings.append("condition metadata is absent from split_info; condition-level reporting cannot be audited")

    return {
        "metrics_path": str(metrics_path),
        "split_info_path": str(metrics_path.parent / "split_info.json"),
        "split_mode": split_mode,
        "split_gap": split_info.get("split_gap"),
        "early_cycle": split_info.get("early_cycle"),
        "horizon": split_info.get("horizon"),
        "fixed_len": split_info.get("fixed_len"),
        "feature_mode": split_info.get("feature_mode"),
        "target_mode": split_info.get("target_mode"),
        "datasets": datasets,
        "test_datasets": sorted({dataset_by_file[name] for name in split_files["test"] if name in dataset_by_file}),
        "excluded_datasets": excluded,
        "unknown_datasets": unknown,
        "train_files": split_files["train"],
        "val_files": split_files["val"],
        "test_files": split_files["test"],
        "missing_files": missing_files,
        "warnings": warnings,
        "failures": failures,
        "policy_ok": not failures,
    }


def resolve_metrics_paths(search_root: Path, metrics_roots: str, output_dir: Path) -> list[Path]:
    roots = parse_csv_list(metrics_roots)
    paths: list[Path] = []
    if roots:
        for item in roots:
            path = Path(item)
            if not path.is_absolute():
                path = search_root / path
            if path.is_dir():
                candidate = path / "model_comparison_metrics.csv"
            else:
                candidate = path
            if not candidate.exists():
                raise FileNotFoundError(f"missing metrics file: {candidate}")
            paths.append(candidate.resolve())
    else:
        output_dir_resolved = output_dir.resolve()
        for path in search_root.rglob("model_comparison_metrics.csv"):
            resolved = path.resolve()
            if output_dir_resolved in resolved.parents:
                continue
            paths.append(resolved)
    return sorted(dict.fromkeys(paths))


def add_rows_from_metrics(
    metrics_path: Path,
    split_audit: dict[str, Any],
    include_models: set[str],
    exclude_models: set[str],
    allow_policy_failures: bool,
    exclude_tuned_outputs: bool,
    require_multi_model_metrics: bool,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(metrics_path)
    if "model" not in frame.columns:
        return []
    learned_models = sorted({str(model) for model in frame["model"] if str(model) not in {"", "persistence"}})
    single_model_metrics_risk = len(learned_models) < 2
    if require_multi_model_metrics and single_model_metrics_risk:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        model = str(row.get("model", ""))
        if include_models and model not in include_models:
            continue
        if model in exclude_models:
            continue
        rmse = finite_or_none(row.get("RMSE"))
        mae = finite_or_none(row.get("MAE"))
        if rmse is None:
            continue
        if not split_audit["policy_ok"] and not allow_policy_failures:
            continue
        tuned_risk, tuned_reason = detect_tuned_hyperparameter_risk(metrics_path, row)
        if tuned_risk and exclude_tuned_outputs:
            continue
        problem_key = "|".join(
            [
                f"horizon={split_audit['horizon']}",
                f"early_cycle={split_audit['early_cycle']}",
                f"fixed_len={split_audit['fixed_len']}",
                f"feature_mode={split_audit['feature_mode']}",
                f"target_mode={split_audit['target_mode']}",
                f"split_mode={split_audit['split_mode']}",
                f"test_datasets={','.join(split_audit['test_datasets'])}",
            ]
        )
        candidate = row.to_dict()
        candidate.update(
            {
                "output_root": str(metrics_path.parent),
                "metrics_path": str(metrics_path),
                "policy_ok": split_audit["policy_ok"],
                "policy_failures": "; ".join(split_audit["failures"]),
                "policy_warnings": "; ".join(split_audit["warnings"]),
                "split_mode": split_audit["split_mode"],
                "split_gap": split_audit["split_gap"],
                "early_cycle": split_audit["early_cycle"],
                "horizon": split_audit["horizon"],
                "fixed_len": split_audit["fixed_len"],
                "feature_mode": split_audit["feature_mode"],
                "target_mode": split_audit["target_mode"],
                "test_datasets": ",".join(split_audit["test_datasets"]),
                "all_datasets": ",".join(split_audit["datasets"]),
                "problem_key": problem_key,
                "tuned_hyperparameter_risk": tuned_risk,
                "tuned_hyperparameter_reason": tuned_reason,
                "single_model_metrics_risk": single_model_metrics_risk,
                "learned_models_in_metrics": ",".join(learned_models),
                "selected_by": "test_RMSE_then_test_MAE",
            }
        )
        if mae is None:
            candidate["MAE"] = float("nan")
        rows.append(candidate)
    return rows


def detect_tuned_hyperparameter_risk(metrics_path: Path, row: pd.Series) -> tuple[bool, str]:
    haystacks = [
        str(metrics_path.parent).lower(),
        str(metrics_path).lower(),
        str(row.get("config_name", "")).lower(),
    ]
    matched = sorted({marker for marker in TUNED_OUTPUT_MARKERS if any(marker in text for text in haystacks)})
    if matched:
        return True, "matched markers: " + ",".join(matched)
    return False, ""


def selection_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str, str]:
    return (
        metric_or_inf(row.get("RMSE")),
        metric_or_inf(row.get("MAE")),
        -metric_or_neg_inf(row.get("R2")),
        str(row.get("model", "")),
        str(row.get("output_root", "")),
    )


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, row in enumerate(sorted(candidates, key=selection_sort_key), start=1):
        item = dict(row)
        item["test_selection_rank"] = rank
        item["selected_final_model"] = rank == 1
        ranked.append(item)
    return ranked


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    columns = [column for column in columns if any(column in row for row in rows)]
    if not columns:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_markdown_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def format_markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(
    output_dir: Path,
    selected: dict[str, Any] | None,
    ranked: list[dict[str, Any]],
    split_audits: list[dict[str, Any]],
    raw_audit: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    problem_keys = sorted({str(row.get("problem_key", "")) for row in ranked if row.get("problem_key")})
    tuned_rows = [row for row in ranked if row.get("tuned_hyperparameter_risk")]
    single_model_rows = [row for row in ranked if row.get("single_model_metrics_risk")]
    lines = [
        "# Final Model Selection By Test Metrics",
        "",
        "Selection rule: lowest test RMSE, then lowest test MAE. Validation metrics are ignored for final selection.",
        "Interpret rankings only within the same problem setting: horizon, early cycle, fixed length, feature mode, target mode, split mode, and test datasets.",
        f"Dataset policy: `{args.dataset_policy}`.",
        f"Exclude tuned outputs: `{args.exclude_tuned_outputs}`.",
        f"Require multi-model metrics: `{args.require_multi_model_metrics}`.",
        f"Raw data files checked: `{raw_audit['n_raw_pkl_files']}`.",
        f"Li-ion datasets present: `{', '.join(raw_audit['present_liion_datasets'])}`.",
    ]
    if len(problem_keys) > 1:
        lines.append(f"Eligible candidates span `{len(problem_keys)}` problem settings; use `--metrics-roots` for a locked final comparison.")
    if tuned_rows:
        lines.append(f"Eligible candidates with tuned-hyperparameter risk: `{len(tuned_rows)}`.")
    if single_model_rows:
        lines.append(f"Eligible candidates from single-learned-model metric files: `{len(single_model_rows)}`.")
    if raw_audit["present_excluded_datasets"]:
        lines.append(f"Excluded non-Li-ion datasets present locally: `{', '.join(raw_audit['present_excluded_datasets'])}`.")
    if raw_audit["missing_liion_datasets"]:
        lines.append(f"Li-ion datasets missing locally: `{', '.join(raw_audit['missing_liion_datasets'])}`.")

    if selected is not None:
        lines.extend(
            [
                "",
                "## Selected Model",
                "",
                f"- Model: `{selected.get('model')}`",
                f"- Test RMSE: `{format_markdown_cell(selected.get('RMSE'))}`",
                f"- Test MAE: `{format_markdown_cell(selected.get('MAE'))}`",
                f"- Test R2: `{format_markdown_cell(selected.get('R2'))}`",
                f"- Output root: `{selected.get('output_root')}`",
                f"- Test datasets: `{selected.get('test_datasets')}`",
                f"- Split mode: `{selected.get('split_mode')}`",
                f"- Horizon: `{selected.get('horizon')}`",
                f"- Checkpoint: `{selected.get('checkpoint_path', '')}`",
            ]
        )
    else:
        lines.extend(["", "## Selected Model", "", "_No eligible model rows were found._"])

    invalid = [audit for audit in split_audits if not audit["policy_ok"]]
    warning = [audit for audit in split_audits if audit["policy_ok"] and audit["warnings"]]
    lines.extend(
        [
            "",
            "## Ranked Eligible Candidates",
            "",
            markdown_table(
                ranked,
                [
                    "test_selection_rank",
                    "model",
                    "RMSE",
                    "MAE",
                    "R2",
                    "horizon",
                    "split_mode",
                    "test_datasets",
                    "problem_key",
                    "tuned_hyperparameter_risk",
                    "tuned_hyperparameter_reason",
                    "single_model_metrics_risk",
                    "learned_models_in_metrics",
                    "output_root",
                    "checkpoint_path",
                ],
                limit=args.report_limit,
            ),
            "",
            "## Split Audit",
            "",
            markdown_table(
                split_audits,
                [
                    "policy_ok",
                    "split_mode",
                    "split_gap",
                    "early_cycle",
                    "horizon",
                    "test_datasets",
                    "excluded_datasets",
                    "failures",
                    "warnings",
                    "metrics_path",
                ],
                limit=args.report_limit,
            ),
        ]
    )
    if invalid:
        lines.append("")
        lines.append(f"Invalid metric roots excluded by policy: `{len(invalid)}`.")
    if warning:
        lines.append(f"Metric roots with audit warnings: `{len(warning)}`.")

    (output_dir / "final_test_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    search_root = Path(args.search_root).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = search_root / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = search_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(search_root)
    raw_audit = audit_raw_samples(data_dir, manifest)
    metrics_paths = resolve_metrics_paths(search_root, args.metrics_roots, output_dir)
    include_models = set(parse_csv_list(args.models))
    exclude_models = set(parse_csv_list(args.exclude_models))

    split_audits: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for metrics_path in metrics_paths:
        split_info = load_split_info(metrics_path)
        audit = audit_split(metrics_path, split_info, manifest, data_dir, args.dataset_policy)
        split_audits.append(audit)
        candidates.extend(
            add_rows_from_metrics(
                metrics_path,
                audit,
                include_models,
                exclude_models,
                args.allow_policy_failures,
                args.exclude_tuned_outputs,
                args.require_multi_model_metrics,
            )
        )

    ranked = rank_candidates(candidates)
    selected = ranked[0] if ranked else None

    pd.DataFrame(raw_audit["files"]).to_csv(output_dir / "local_dataset_audit.csv", index=False)
    pd.DataFrame(split_audits).to_csv(output_dir / "metrics_split_audit.csv", index=False)
    pd.DataFrame(ranked).to_csv(output_dir / "final_test_model_candidates.csv", index=False)

    metadata = {
        "selection_rule": "lowest test RMSE, then lowest test MAE; validation metrics are ignored",
        "dataset_policy": args.dataset_policy,
        "search_root": str(search_root),
        "data_dir": str(data_dir),
        "metrics_paths": [str(path) for path in metrics_paths],
        "raw_dataset_audit": raw_audit,
        "split_audits": split_audits,
        "selected": selected,
    }
    (output_dir / "final_test_model_selection.json").write_text(
        json.dumps(clean_json_value(metadata), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    write_report(output_dir, selected, ranked, split_audits, raw_audit, args)

    if selected is None:
        print("No eligible model rows found.")
    else:
        print(f"selected_model={selected.get('model')}")
        print(f"test_RMSE={selected.get('RMSE')} test_MAE={selected.get('MAE')}")
        print(f"output_root={selected.get('output_root')}")
    print(f"candidates={output_dir / 'final_test_model_candidates.csv'}")
    print(f"report={output_dir / 'final_test_model_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", default=".")
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument(
        "--metrics-roots",
        default="",
        help=(
            "Comma-separated output dirs or model_comparison_metrics.csv files. "
            "When omitted, recursively scans --search-root."
        ),
    )
    parser.add_argument("--output-dir", default="final_model_test_selection")
    parser.add_argument("--dataset-policy", choices=["liion_only", "all"], default="liion_only")
    parser.add_argument("--models", default="", help="Optional comma-separated model allow-list.")
    parser.add_argument("--exclude-models", default=DEFAULT_EXCLUDE_MODELS)
    parser.add_argument(
        "--allow-policy-failures",
        action="store_true",
        help="Include metric rows even when split/dataset usage violates the dataset plan.",
    )
    parser.add_argument(
        "--exclude-tuned-outputs",
        action="store_true",
        help="Exclude rows from output roots/config names that look like tuned, Optuna, refinement, or locked-test artifacts.",
    )
    parser.add_argument(
        "--require-multi-model-metrics",
        action="store_true",
        help="Only include metrics files containing at least two learned models, excluding persistence.",
    )
    parser.add_argument("--report-limit", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
