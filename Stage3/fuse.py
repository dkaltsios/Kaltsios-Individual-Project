"""
Weighted linear pool late fusion: p* ∝ w * p_stage1 + (1-w) * p_stage2, renormalized.
Merge prediction CSVs on sample_id (from Stage1/eval.py and Stage2/eval.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage2.eval import evaluate_multiclass, evaluate_probs


def weighted_linear_pool(p1: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    raw = w * p1 + (1.0 - w) * p2
    s = raw.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    return raw / s


def _prob_columns(df: pd.DataFrame) -> List[str]:
    """Preserve CSV column order (matches class index 0..K-1 from Stage1/2 eval)."""
    return [c for c in df.columns if c.startswith("prob_")]


def _load_and_merge(
    path1: Path, path2: Path
) -> Tuple[pd.DataFrame, List[str]]:
    s1 = pd.read_csv(path1)
    s2 = pd.read_csv(path2)
    if "sample_id" not in s1.columns or "sample_id" not in s2.columns:
        raise SystemExit(
            "Both prediction CSVs must include a sample_id column. "
            "Re-run Stage1/eval.py and Stage2/eval.py with the updated scripts."
        )
    s1["sample_id"] = s1["sample_id"].astype(str)
    s2["sample_id"] = s2["sample_id"].astype(str)
    cols1 = _prob_columns(s1)
    cols2 = _prob_columns(s2)
    if not cols1 or not cols2:
        return (
            pd.merge(s1, s2, on="sample_id", how="inner", suffixes=("_s1", "_s2")),
            [],
        )
    if cols1 != cols2:
        raise SystemExit(
            f"prob_* columns differ between inputs: {cols1!r} vs {cols2!r}"
        )
    merged = pd.merge(s1, s2, on="sample_id", how="inner", suffixes=("_s1", "_s2"))
    if merged.empty:
        raise SystemExit("No overlapping sample_id rows after merge.")
    return merged, cols1


def _assert_y_agreement(merged: pd.DataFrame) -> np.ndarray:
    if "y_true_s1" not in merged.columns or "y_true_s2" not in merged.columns:
        raise SystemExit("Merged frame missing y_true_s1 / y_true_s2.")
    if not (merged["y_true_s1"].values == merged["y_true_s2"].values).all():
        raise SystemExit("y_true disagrees between Stage1 and Stage2 for some sample_id.")
    return merged["y_true_s1"].astype(int).to_numpy()


def _tune_w_val(
    val_merged: pd.DataFrame,
    prob_cols: List[str],
    grid_steps: int,
) -> Tuple[float, float]:
    p1 = val_merged[[f"{c}_s1" for c in prob_cols]].to_numpy(dtype=np.float64)
    p2 = val_merged[[f"{c}_s2" for c in prob_cols]].to_numpy(dtype=np.float64)
    y_true = _assert_y_agreement(val_merged)
    num_classes = p1.shape[1]
    best_w, best_f1 = 0.0, -1.0
    for w in np.linspace(0.0, 1.0, grid_steps):
        fused = weighted_linear_pool(p1, p2, float(w))
        y_pred = fused.argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    return best_w, best_f1


def _fuse_binary(merged: pd.DataFrame, w: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1 = merged["y_prob_s1"].to_numpy(dtype=np.float64)
    p2 = merged["y_prob_s2"].to_numpy(dtype=np.float64)
    y_prob = w * p1 + (1.0 - w) * p2
    y_true = _assert_y_agreement(merged)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred


def _fuse_multiclass(
    merged: pd.DataFrame, prob_cols: List[str], w: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1 = merged[[f"{c}_s1" for c in prob_cols]].to_numpy(dtype=np.float64)
    p2 = merged[[f"{c}_s2" for c in prob_cols]].to_numpy(dtype=np.float64)
    y_prob = weighted_linear_pool(p1, p2, w)
    y_true = _assert_y_agreement(merged)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_prob, y_pred


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _stack_features_multiclass(merged: pd.DataFrame, prob_cols: List[str]) -> np.ndarray:
    p1 = merged[[f"{c}_s1" for c in prob_cols]].to_numpy(dtype=np.float64)
    p2 = merged[[f"{c}_s2" for c in prob_cols]].to_numpy(dtype=np.float64)
    return np.concatenate([p1, p2], axis=1)


def _fit_stacking_multiclass(
    val_merged: pd.DataFrame, prob_cols: List[str], c_value: float, max_iter: int
) -> LogisticRegression:
    x_val = _stack_features_multiclass(val_merged, prob_cols)
    y_val = _assert_y_agreement(val_merged)
    # Keep compatibility with older sklearn versions where some kwargs differ.
    try:
        clf = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
    except TypeError:
        clf = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
        )
    clf.fit(x_val, y_val)
    return clf


def _fit_stacking_binary(
    val_merged: pd.DataFrame, c_value: float, max_iter: int
) -> LogisticRegression:
    x_val = np.column_stack(
        [
            val_merged["y_prob_s1"].to_numpy(dtype=np.float64),
            val_merged["y_prob_s2"].to_numpy(dtype=np.float64),
        ]
    )
    y_val = _assert_y_agreement(val_merged)
    try:
        clf = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )
    except TypeError:
        clf = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
        )
    clf.fit(x_val, y_val)
    return clf


def main():
    parser = argparse.ArgumentParser(description="Stage 3 late fusion")
    parser.add_argument("--stage1_preds", type=Path, required=True, help="Stage 1 test_predictions.csv")
    parser.add_argument("--stage2_preds", type=Path, required=True, help="Stage 2 test_predictions.csv")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--method",
        choices=("weighted_pool", "stacking_lr"),
        default="weighted_pool",
        help="Late fusion method: weighted probability pool or logistic-regression stacking.",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=None,
        help="Weight on Stage 1 probabilities in [0,1]. If omitted, use 0.5 unless val preds tune it.",
    )
    parser.add_argument("--val_stage1_preds", type=Path, default=None)
    parser.add_argument("--val_stage2_preds", type=Path, default=None)
    parser.add_argument(
        "--grid_steps",
        type=int,
        default=21,
        help="Grid size for w in [0,1] when tuning on validation predictions.",
    )
    parser.add_argument(
        "--label_mapping",
        type=Path,
        default=None,
        help="JSON with class_names (multiclass); default class indices if omitted.",
    )
    parser.add_argument(
        "--binary_threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary stacking predictions.",
    )
    parser.add_argument(
        "--stacking_c",
        type=float,
        default=1.0,
        help="Inverse regularization strength C for logistic-regression stacking.",
    )
    parser.add_argument(
        "--stacking_max_iter",
        type=int,
        default=2000,
        help="Max iterations for logistic-regression stacking fit.",
    )
    args = parser.parse_args()

    label_mapping_path = args.label_mapping
    if label_mapping_path is None:
        cand = PROJECT_ROOT / "Dataset" / "label_mapping_multiclass.json"
        if cand.exists():
            label_mapping_path = cand

    merged, prob_cols = _load_and_merge(args.stage1_preds, args.stage2_preds)
    tuned_on_val = False
    val_macro_f1: Optional[float] = None
    val_acc_tuned: Optional[float] = None
    stacker_trained_on_val = False

    w = args.w
    if w is not None and (w < 0.0 or w > 1.0):
        raise SystemExit("--w must be in [0, 1]")

    if prob_cols:
        # Multiclass branch.
        if args.method == "stacking_lr":
            if args.val_stage1_preds is None or args.val_stage2_preds is None:
                raise SystemExit(
                    "stacking_lr requires --val_stage1_preds and --val_stage2_preds."
                )
            val_merged, val_prob_cols = _load_and_merge(
                args.val_stage1_preds, args.val_stage2_preds
            )
            if val_prob_cols != prob_cols:
                raise SystemExit("Validation prob columns must match test prob columns.")
            stacker = _fit_stacking_multiclass(
                val_merged, prob_cols, args.stacking_c, args.stacking_max_iter
            )
            stacker_trained_on_val = True

            class_names = None
            num_classes = len(prob_cols)
            if label_mapping_path and Path(label_mapping_path).exists():
                with open(Path(label_mapping_path)) as f:
                    lm = json.load(f)
                class_names = lm.get("class_names", list(range(num_classes)))
            if class_names is None:
                class_names = [str(i) for i in range(num_classes)]

            x_test = _stack_features_multiclass(merged, prob_cols)
            y_prob = stacker.predict_proba(x_test)
            y_true = _assert_y_agreement(merged)
            y_pred = y_prob.argmax(axis=1)
            _, _, _, metrics = evaluate_multiclass(y_true, y_prob, num_classes, class_names)
            metrics_save = _to_json_safe(metrics)

            pred_names = [class_names[int(p)] for p in y_pred]
            true_names = [class_names[int(t)] for t in y_true]
            out_df = pd.DataFrame(
                {
                    "sample_id": merged["sample_id"].astype(str).values,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "true_class": true_names,
                    "predicted_class": pred_names,
                }
            )
            for c, name in enumerate(class_names):
                out_df[f"prob_{name}"] = y_prob[:, c]
        else:
            # Weighted pool (current method).
            if args.val_stage1_preds is not None and args.val_stage2_preds is not None:
                if w is None:
                    val_merged, val_prob_cols = _load_and_merge(
                        args.val_stage1_preds, args.val_stage2_preds
                    )
                    if val_prob_cols != prob_cols:
                        raise SystemExit("Validation prob columns must match test prob columns.")
                    w, val_f1 = _tune_w_val(val_merged, prob_cols, args.grid_steps)
                    tuned_on_val = True
                    val_macro_f1 = val_f1
                    print(json.dumps({"tuned_w": w, "val_macro_f1": val_f1}, indent=2))
                else:
                    print("Note: --w is set; skipping validation tuning.")
            if w is None:
                w = 0.5

            class_names = None
            num_classes = len(prob_cols)
            if label_mapping_path and Path(label_mapping_path).exists():
                with open(Path(label_mapping_path)) as f:
                    lm = json.load(f)
                class_names = lm.get("class_names", list(range(num_classes)))
            if class_names is None:
                class_names = [str(i) for i in range(num_classes)]

            y_true, y_prob, y_pred = _fuse_multiclass(merged, prob_cols, w)
            _, _, _, metrics = evaluate_multiclass(y_true, y_prob, num_classes, class_names)
            metrics_save = _to_json_safe(metrics)

            pred_names = [class_names[int(p)] for p in y_pred]
            true_names = [class_names[int(t)] for t in y_true]
            out_df = pd.DataFrame(
                {
                    "sample_id": merged["sample_id"].astype(str).values,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "true_class": true_names,
                    "predicted_class": pred_names,
                }
            )
            for c, name in enumerate(class_names):
                out_df[f"prob_{name}"] = y_prob[:, c]
    else:
        # Binary branch.
        if args.binary_threshold < 0.0 or args.binary_threshold > 1.0:
            raise SystemExit("--binary_threshold must be in [0,1]")
        if args.method == "stacking_lr":
            if args.val_stage1_preds is None or args.val_stage2_preds is None:
                raise SystemExit(
                    "stacking_lr requires --val_stage1_preds and --val_stage2_preds."
                )
            val_merged, _ = _load_and_merge(args.val_stage1_preds, args.val_stage2_preds)
            stacker = _fit_stacking_binary(val_merged, args.stacking_c, args.stacking_max_iter)
            stacker_trained_on_val = True
            x_test = np.column_stack(
                [
                    merged["y_prob_s1"].to_numpy(dtype=np.float64),
                    merged["y_prob_s2"].to_numpy(dtype=np.float64),
                ]
            )
            y_true = _assert_y_agreement(merged)
            y_prob = stacker.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= args.binary_threshold).astype(int)
            _, _, _, metrics = evaluate_probs(y_true, y_prob)
            metrics_save = metrics
            out_df = pd.DataFrame(
                {
                    "sample_id": merged["sample_id"].astype(str).values,
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                }
            )
        else:
            if w is None:
                if args.val_stage1_preds and args.val_stage2_preds:
                    vm, _ = _load_and_merge(args.val_stage1_preds, args.val_stage2_preds)
                    best_w, best_acc = 0.0, -1.0
                    p1v = vm["y_prob_s1"].to_numpy()
                    p2v = vm["y_prob_s2"].to_numpy()
                    yv = _assert_y_agreement(vm)
                    for ww in np.linspace(0.0, 1.0, args.grid_steps):
                        pr = ww * p1v + (1.0 - ww) * p2v
                        pred = (pr >= 0.5).astype(int)
                        acc = accuracy_score(yv, pred)
                        if acc > best_acc:
                            best_acc, best_w = acc, float(ww)
                    w = best_w
                    tuned_on_val = True
                    val_acc_tuned = best_acc
                    print(json.dumps({"tuned_w": w, "val_acc": best_acc}, indent=2))
                else:
                    w = 0.5
            y_true, y_prob, y_pred = _fuse_binary(merged, w)
            _, _, _, metrics = evaluate_probs(y_true, y_prob)
            metrics_save = metrics
            out_df = pd.DataFrame(
                {
                    "sample_id": merged["sample_id"].astype(str).values,
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                }
            )

    print(metrics_save)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "test_metrics.json").write_text(json.dumps(metrics_save, indent=2))
    out_df.to_csv(args.out_dir / "test_predictions.csv", index=False)
    fusion_meta = {
        "method": args.method,
        "w": w,
        "tuned_on_val": tuned_on_val,
        "stage1_preds": str(args.stage1_preds),
        "stage2_preds": str(args.stage2_preds),
    }
    if args.method == "stacking_lr":
        fusion_meta["stacking_c"] = args.stacking_c
        fusion_meta["stacking_max_iter"] = args.stacking_max_iter
        fusion_meta["stacker_trained_on_val"] = stacker_trained_on_val
        if not prob_cols:
            fusion_meta["binary_threshold"] = args.binary_threshold
    if args.val_stage1_preds is not None:
        fusion_meta["val_stage1_preds"] = str(args.val_stage1_preds)
    if args.val_stage2_preds is not None:
        fusion_meta["val_stage2_preds"] = str(args.val_stage2_preds)
    if tuned_on_val:
        fusion_meta["grid_steps"] = args.grid_steps
        if val_macro_f1 is not None:
            fusion_meta["val_macro_f1"] = val_macro_f1
            fusion_meta["tuning_objective"] = "macro_f1"
        if val_acc_tuned is not None:
            fusion_meta["val_accuracy"] = val_acc_tuned
            fusion_meta["tuning_objective"] = "accuracy"
    if not tuned_on_val and args.w is None and (
        args.val_stage1_preds is None or args.val_stage2_preds is None
    ):
        fusion_meta["note"] = (
            "w defaulted to 0.5 (no val tuning). "
            "Pass --val_stage1_preds and --val_stage2_preds without --w to tune w on validation."
        )
    (args.out_dir / "fusion_meta.json").write_text(json.dumps(fusion_meta, indent=2))


if __name__ == "__main__":
    main()
