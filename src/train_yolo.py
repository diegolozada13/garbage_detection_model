import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import YOLO_DIR, RUNS_DIR


def train_yolo(model_name, epochs, imgsz, batch, run_name, device):
    data_yaml = YOLO_DIR / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"No existe {data_yaml}. Ejecuta primero: py -m src.export_yolo"
        )

    model = YOLO(model_name)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(RUNS_DIR),
        name=run_name,
        device=device,
    )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="yolo_baseline")
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    train_yolo(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()