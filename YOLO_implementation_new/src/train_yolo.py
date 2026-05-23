import argparse
from pathlib import Path
from ultralytics import YOLO

from src.config import DATASET_YAML, RUNS_DIR


def train_yolo(model_name, epochs, imgsz, batch, run_name, device, use_aug, data=None):

    # if not data_yaml.exists():
    #     raise FileNotFoundError(
    #         f"No existe {data_yaml}. Ejecuta primero: py -m src.export_yolo"
    #     )

    model = YOLO(model_name)

    train_args = {
        "data": str(DATASET_YAML),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(RUNS_DIR),
        "name": run_name,
        "device": device,
    }

    if use_aug:
        train_args.update({
            "hsv_h": 0.005,
            "hsv_s": 0.25,
            "hsv_v": 0.20,
            "translate": 0.0,
            "scale": 0.10,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
        })

    results = model.train(**train_args)
    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="yolo_baseline")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    train_yolo(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.name,
        device=args.device,
        use_aug=args.aug,
        data=args.data
    )


if __name__ == "__main__":
    main()