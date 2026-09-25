from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the compact SHAP-guided RBF-KAN reference workflow on tabular spectra."
    )
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path)
    parser.add_argument("--target", required=True, help="Target-column name")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--preprocessing",
        nargs="*",
        default=[],
        choices=["bc", "snv", "sg", "d1", "d2"],
    )
    parser.add_argument("--hidden", nargs="+", type=int, default=[16])
    parser.add_argument("--num-basis", type=int, default=5)
    parser.add_argument("--grid-half-width", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-epochs", type=int, default=700)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("reference_summary.json"))
    return parser


def _read_table(path: Path, target: str, feature_columns: list[str] | None = None):
    frame = pd.read_csv(path)
    if target not in frame:
        raise ValueError(f"target column '{target}' is absent from {path}")
    if feature_columns is None:
        feature_columns = [
            name
            for name in frame.select_dtypes(include=[np.number]).columns
            if name != target
        ]
        if not feature_columns:
            raise ValueError("no numeric feature columns were found")
    missing = [name for name in feature_columns if name not in frame]
    if missing:
        raise ValueError(f"prediction table is missing feature columns: {missing}")
    try:
        features = frame[feature_columns].to_numpy(dtype=float)
        targets = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("features and target values must be numeric") from error
    if not np.isfinite(features).all():
        raise ValueError("feature values must be finite")
    if not np.isfinite(targets).all():
        raise ValueError("target values must be finite")
    return (
        features,
        targets,
        feature_columns,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from shap_rbf_kan.preprocessing import apply_pipeline
    from shap_rbf_kan.workflow import run_shap_guided_workflow

    calibration_x, calibration_y, feature_columns = _read_table(
        args.calibration_csv, args.target
    )
    calibration_x = apply_pipeline(calibration_x, args.preprocessing)
    prediction_x = None
    prediction_y = None
    if args.prediction_csv:
        prediction_x, prediction_y, _ = _read_table(
            args.prediction_csv, args.target, feature_columns
        )
        prediction_x = apply_pipeline(prediction_x, args.preprocessing)

    config = {
        "hidden_features": args.hidden,
        "num_basis": args.num_basis,
        "grid_half_width": args.grid_half_width,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    result = run_shap_guided_workflow(
        calibration_x,
        calibration_y,
        config=config,
        top_k=args.top_k,
        prediction_x=prediction_x,
        prediction_y=prediction_y,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
    )
    selected_columns = [feature_columns[index] for index in result.selected_features]
    payload = {
        "selected_feature_indices": result.selected_features.tolist(),
        "model_feature_indices": result.model_feature_indices.tolist(),
        "selected_feature_names": selected_columns,
        "full_selected_epoch": result.full_model.selected_epoch,
        "compact_selected_epoch": result.compact_model.selected_epoch,
        "full_prediction_metrics": result.full_metrics,
        "compact_prediction_metrics": result.compact_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
