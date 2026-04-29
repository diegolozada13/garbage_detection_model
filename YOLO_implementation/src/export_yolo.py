import json
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from src.config import COMMON_DIR, YOLO_DIR, CLASS_NAME_TO_IDX


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return x_center, y_center, w_norm, h_norm


def make_unique_name(file_name):
    return file_name.replace("/", "__").replace("\\", "__")


def load_common_dataset():
    common_path = COMMON_DIR / "common_dataset.json"

    if not common_path.exists():
        raise FileNotFoundError(
            f"No existe {common_path}. Ejecuta primero: py -m src.data_common"
        )

    with open(common_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reset_yolo_dir():
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    for split in ["train", "val", "test"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def export_split(entries, split_name):
    for entry in tqdm(entries, desc=f"Exportando {split_name}"):
        src_img = Path(entry["img_path"])
        unique_name = make_unique_name(entry["file_name"])

        dst_img = YOLO_DIR / "images" / split_name / unique_name
        shutil.copy(src_img, dst_img)

        label_name = Path(unique_name).stem + ".txt"
        label_path = YOLO_DIR / "labels" / split_name / label_name

        lines = []

        for ann in entry["annotations"]:
            class_id = ann["class_id"]
            bbox = ann["bbox_coco"]

            x_center, y_center, w_norm, h_norm = coco_bbox_to_yolo(
                bbox,
                entry["width"],
                entry["height"],
            )

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def create_data_yaml():
    names = {idx: name for name, idx in CLASS_NAME_TO_IDX.items()}

    data = {
        "path": str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }

    yaml_path = YOLO_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    return yaml_path


def count_images(path):
    valid_exts = {".jpg", ".jpeg", ".png"}
    return sum(
        1
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )


def verify_export():
    for split in ["train", "val", "test"]:
        img_dir = YOLO_DIR / "images" / split
        label_dir = YOLO_DIR / "labels" / split

        num_imgs = count_images(img_dir)
        num_labels = len(list(label_dir.glob("*.txt")))

        print(f"{split}: imágenes={num_imgs} | labels={num_labels}")


def main():
    common = load_common_dataset()
    splits = common["splits"]

    reset_yolo_dir()

    for split_name, entries in splits.items():
        export_split(entries, split_name)

    yaml_path = create_data_yaml()

    print("\nExportación YOLO completada.")
    print("data.yaml:", yaml_path)
    verify_export()


if __name__ == "__main__":
    main()