# make_demo_predictions.py
import argparse
import os
import time
import warnings
import pandas as pd

# Quiet the nested-tensor warning from torch's Transformer encoder
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False.*"
)

from demo_predictions import predict_dataframe


def band(p: float) -> str:
    if p >= 0.80: return "very-high"
    if p >= 0.60: return "high"
    if p >= 0.40: return "moderate"
    if p >= 0.20: return "low"
    return "very-low"


def _default_out_path(csv_path: str) -> str:
    base, ext = os.path.splitext(csv_path)
    return f"{base}_scored.csv"


def run_predictions(
    csv: str = "data/new_patients.csv",
    out: str | None = None,
    header: str = "none",          # "none" or "infer"
    threshold: float | None = None,
    show_head: int = 10,
    show_all: bool = False,
) -> pd.DataFrame:
    """
    Load patients CSV, score with demo_predictions.predict_dataframe,
    print timing + preview, and write a scored CSV.

    Returns the scored DataFrame.
    """
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Input CSV not found: {csv}")

    # 1) Load and normalize column names to strings "0","1",...
    hdr = None if header == "none" else "infer"
    df = pd.read_csv(csv, header=hdr)
    df.columns = [str(i) for i in range(df.shape[1])]

    # 2) Time the prediction call (keeps demo_predictions.py untouched)
    t0 = time.perf_counter()
    scored = predict_dataframe(df)   # expects/returns prob_meta & pred_meta
    dt = time.perf_counter() - t0

    # 3) Optional threshold override
    if threshold is not None:
        if "prob_meta" not in scored.columns:
            raise ValueError("predict_dataframe did not return 'prob_meta'.")
        scored["pred_meta"] = (scored["prob_meta"] >= float(threshold)).astype(int)

    # 4) Summary + preview
    n = len(scored)
    rps = (n / dt) if dt > 0 else float("inf")
    print(f"[make_demo_predictions] Predicted {n} patients in {dt:.3f}s ({rps:.1f} rows/s)")

    preview = scored[["prob_meta", "pred_meta"]].copy()
    preview["risk_band"] = preview["prob_meta"].apply(band)

    if show_all or n <= show_head:
        print(preview.to_string(index=False))
    else:
        print(preview.head(show_head).to_string(index=False))
        print(f"... (showing first {show_head} of {n}; use --show-all to print everything)")

    # 5) Keep inputs + predictions together; add risk band; write out
    out_df = df.copy()
    out_df["prob_meta"] = scored["prob_meta"].values
    out_df["pred_meta"] = scored["pred_meta"].values
    out_df["risk_band"] = out_df["prob_meta"].apply(band)


    out_path = out or _default_out_path(csv)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(out_path, index=False)
    print(f"[make_demo_predictions] Wrote scored results -> {out_path}")

    return out_df


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run QISICGM demo predictions on a CSV (with timing) and save a scored file."
    )
    ap.add_argument("--csv", default="data/new_patients.csv", help="Path to input CSV (default: data/new_patients.csv).")
    ap.add_argument("--out", default="", help="Optional path for the scored CSV (default: <csv>_scored.csv).")
    ap.add_argument("--header", choices=["none", "infer"], default="none", help="CSV header handling (default: none).")
    ap.add_argument("--threshold", type=float, default=None, help="Optional meta threshold override.")
    ap.add_argument("--show-head", type=int, default=10, help="Rows to preview when not showing all (default: 10).")
    ap.add_argument("--show-all", action="store_true", help="Print all rows to the console.")
    return ap.parse_args()


if __name__ == "__main__":
    # Works with or without CLI args:
    # - With args: `python make_demo_predictions.py --csv path.csv`
    # - Without args: defaults to data/new_patients.csv
    args = _parse_args()
    run_predictions(
        csv=args.csv,
        out=(args.out or None),
        header=args.header,
        threshold=args.threshold,
        show_head=args.show_head,
        show_all=args.show_all,
    )
