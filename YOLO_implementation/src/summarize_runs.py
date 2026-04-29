from pathlib import Path
import pandas as pd

from src.config import RUNS_DIR


def find_results_csv():
    return sorted(RUNS_DIR.rglob("results.csv"))


def summarize_run(results_csv):
    run_dir = results_csv.parent
    df = pd.read_csv(results_csv)

    # limpiar espacios en columnas
    df.columns = [c.strip() for c in df.columns]

    last = df.iloc[-1]
    best_map50_idx = df["metrics/mAP50(B)"].idxmax()
    best = df.loc[best_map50_idx]

    return {
        "run": run_dir.name,
        "epochs": int(last["epoch"]),
        "best_epoch_mAP50": int(best["epoch"]),
        "precision": round(float(best["metrics/precision(B)"]), 4),
        "recall": round(float(best["metrics/recall(B)"]), 4),
        "mAP50": round(float(best["metrics/mAP50(B)"]), 4),
        "mAP50-95": round(float(best["metrics/mAP50-95(B)"]), 4),
        "box_loss": round(float(best["val/box_loss"]), 4),
        "cls_loss": round(float(best["val/cls_loss"]), 4),
        "dfl_loss": round(float(best["val/dfl_loss"]), 4),
        "path": str(run_dir),
    }


def main():
    result_files = find_results_csv()

    if not result_files:
        print(f"No se encontraron results.csv en {RUNS_DIR}")
        return

    rows = [summarize_run(csv) for csv in result_files]
    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=["mAP50", "mAP50-95"],
        ascending=False
    )

    out_csv = RUNS_DIR / "summary_results.csv"
    out_md = RUNS_DIR / "summary_results.md"

    summary.to_csv(out_csv, index=False, encoding="utf-8")
    summary.to_markdown(out_md, index=False)

    print("\nResumen de experimentos:\n")
    print(summary.to_markdown(index=False))

    print("\nArchivos guardados:")
    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()