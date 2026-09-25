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
        description=(
            "Tune one regression model with calibration-only five-fold Optuna-TPE "
            "optimization and save a resumable study plus complete trial records."
        )
    )
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path)
    parser.add_argument("--target", required=True, help="Target-column name")
    parser.add_argument(
        "--model",
        required=True,
        choices=["svr", "xgboost", "mlp", "1d-cnn", "b-spline-kan"],
    )
    parser.add_argument("--dataset", choices=["fruit", "oil"], default="fruit")
    parser.add_argument(
        "--preprocessing",
        nargs="*",
        default=[],
        choices=["bc", "snv", "sg", "d1", "d2"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--trials",
        type=int,
        help="Total completed-trial target; defaults to the declared model budget.",
    )
    parser.add_argument("--study-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("bayesian_search_output"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--mlp-max-epochs", type=int)
    parser.add_argument("--mlp-patience", type=int)
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
        raise ValueError(f"table is missing feature columns: {missing}")
    try:
        features = frame[feature_columns].to_numpy(dtype=float)
        targets = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("features and target values must be numeric") from error
    if not np.isfinite(features).all():
        raise ValueError("feature values must be finite")
    if not np.isfinite(targets).all():
        raise ValueError("target values must be finite")
    return features, targets, feature_columns


def _regression_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    residual = prediction - reference
    total = float(np.sum((reference - reference.mean()) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def _safe_prefix(dataset: str, model: str) -> str:
    return f"{dataset}_{model}".replace("-", "_")


def _write_trials(study, path: Path) -> None:
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        row: dict[str, object] = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "mean_rmse": trial.value,
            "sd_rmse": trial.user_attrs.get("sd_rmse"),
            "fold_rmse": json.dumps(trial.user_attrs.get("fold_rmse", [])),
        }
        row.update({f"param_{name}": value for name, value in trial.params.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from shap_rbf_kan.model_search import (
        fit_final_model,
        make_fit_predict,
        training_protocol,
    )
    from shap_rbf_kan.preprocessing import apply_pipeline
    from shap_rbf_kan.tuning import (
        SEARCH_BUDGETS,
        bayesian_search,
        decode_hyperparameters,
        select_best_trial,
    )

    calibration_x, calibration_y, feature_columns = _read_table(
        args.calibration_csv, args.target
    )
    calibration_x = apply_pipeline(calibration_x, args.preprocessing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = (args.output_dir / "optuna_studies.sqlite3").resolve()
    storage = f"sqlite:///{database_path.as_posix()}"
    study_name = args.study_name or f"{args.dataset}-{args.model}-tpe-v1"
    target_trials = (
        args.trials if args.trials is not None else SEARCH_BUDGETS[args.model]
    )
    fit_predict = make_fit_predict(
        args.model,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        mlp_max_epochs=args.mlp_max_epochs,
        mlp_patience=args.mlp_patience,
        device=args.device,
    )
    study = bayesian_search(
        calibration_x,
        calibration_y,
        model=args.model,
        dataset=args.dataset,
        fit_predict=fit_predict,
        n_trials=target_trials,
        folds=args.folds,
        seed=args.seed,
        storage=storage,
        study_name=study_name,
        load_if_exists=args.resume,
        protocol_metadata={
            "feature_columns": feature_columns,
            "preprocessing": args.preprocessing,
            "training": training_protocol(
                args.model,
                max_epochs=args.max_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                mlp_max_epochs=args.mlp_max_epochs,
                mlp_patience=args.mlp_patience,
                device=args.device,
            ),
        },
    )
    best = select_best_trial(study)
    best_parameters = decode_hyperparameters(args.model, best.params)
    fitted = fit_final_model(
        args.model,
        calibration_x,
        calibration_y,
        best_parameters,
        seed=args.seed,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        mlp_max_epochs=args.mlp_max_epochs,
        mlp_patience=args.mlp_patience,
        device=args.device,
    )

    prediction_metrics = None
    if args.prediction_csv is not None:
        prediction_x, prediction_y, _ = _read_table(
            args.prediction_csv, args.target, feature_columns
        )
        prediction_x = apply_pipeline(prediction_x, args.preprocessing)
        prediction_metrics = _regression_metrics(prediction_y, fitted.predict(prediction_x))

    completed_trials = sum(
        trial.state.name == "COMPLETE" for trial in study.trials
    )
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "feature_count": int(calibration_x.shape[1]),
        "preprocessing": args.preprocessing,
        "search": {
            "sampler": "TPESampler",
            "sampler_seed": args.seed,
            "objective": "minimum calibration-only K-fold mean RMSE",
            "folds": args.folds,
            "pruning": "disabled",
            "stopping_rule": "complete the declared trial budget",
            "completed_trials": completed_trials,
            "target_trials": target_trials,
            "study_name": study.study_name,
            "optuna_version": study.user_attrs["optuna_version"],
            "search_space_version": study.user_attrs["search_space_version"],
        },
        "search_space": {
            name: str(distribution) for name, distribution in best.distributions.items()
        },
        "best_trial": {
            "number": best.number,
            "mean_rmse": float(best.value),
            "sd_rmse": float(best.user_attrs["sd_rmse"]),
            "fold_rmse": best.user_attrs["fold_rmse"],
            "parameters": best_parameters,
        },
        "final_selected_epoch": fitted.selected_epoch,
        "prediction_metrics": prediction_metrics,
    }
    prefix = _safe_prefix(args.dataset, args.model)
    summary_path = args.output_dir / f"{prefix}_summary.json"
    trials_path = args.output_dir / f"{prefix}_trials.csv"
    _write_trials(study, trials_path)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
