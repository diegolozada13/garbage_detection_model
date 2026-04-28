from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RUNS_DIR


def plot_run(run_name):
    run_dir = RUNS_DIR / run_name
    results_csv = run_dir / "results.csv"

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    epochs = df["epoch"]

    # 📉 LOSS
    plt.figure()
    plt.plot(epochs, df["train/box_loss"], label="train box")
    plt.plot(epochs, df["val/box_loss"], label="val box")
    plt.plot(epochs, df["train/cls_loss"], label="train cls")
    plt.plot(epochs, df["val/cls_loss"], label="val cls")
    plt.plot(epochs, df["train/dfl_loss"], label="train dfl")
    plt.plot(epochs, df["val/dfl_loss"], label="val dfl")
    plt.legend()
    plt.title(f"Losses - {run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(run_dir / "loss_plot.png")
    plt.close()

    # 📊 METRICS
    plt.figure()
    plt.plot(epochs, df["metrics/precision(B)"], label="Precision")
    plt.plot(epochs, df["metrics/recall(B)"], label="Recall")
    plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP50")
    plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95")
    plt.legend()
    plt.title(f"Metrics - {run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.savefig(run_dir / "metrics_plot.png")
    plt.close()

    print(f"Gráficas guardadas en: {run_dir}")


if __name__ == "__main__":
    plot_run("yolo8n_oversampling_soft_20")