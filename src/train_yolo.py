import argparse

from ultralytics import YOLO

from src.config import YOLO_DIR, RUNS_DIR


def train_yolo(model_name, epochs, imgsz, batch, run_name, device, use_aug):
    data_yaml = YOLO_DIR / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"No existe {data_yaml}. Ejecuta primero: py -m src.export_yolo"
        )

    model = YOLO(model_name)

    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(RUNS_DIR),
        "name": run_name,
        "device": device,
    }

    if use_aug:
        train_args.update({
            "hsv_h": 0.01,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
            "translate": 0.05,
            "scale": 0.2,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.05,
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

    args = parser.parse_args()

    train_yolo(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.name,
        device=args.device,
        use_aug=args.aug,
    )


if __name__ == "__main__":
    main()