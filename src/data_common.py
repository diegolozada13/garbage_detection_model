import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    ANNOTATIONS_JSON,
    IMAGES_DIR,
    COMMON_DIR,
    GROUP_MAP,
    CLASS_NAME_TO_IDX,
)


def load_coco_annotations():
    with open(ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def build_common_entries(test_size=0.2, val_size=0.5, random_state=42):
    coco = load_coco_annotations()

    categories = coco["categories"]
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    images = coco["images"]
    annotations = coco["annotations"]

    img_id_to_info = {img["id"]: img for img in images}
    img_id_to_anns = defaultdict(list)

    for ann in annotations:
        img_id_to_anns[ann["image_id"]].append(ann)

    dataset_entries = []

    skipped_missing_img = 0
    skipped_no_map = 0
    skipped_invalid_bbox = 0

    for img_id, img_info in img_id_to_info.items():
        file_name = img_info["file_name"]
        img_path = IMAGES_DIR / file_name

        if not img_path.exists():
            skipped_missing_img += 1
            continue

        entry_annotations = []

        for ann in img_id_to_anns.get(img_id, []):
            cat_name = cat_id_to_name[ann["category_id"]]

            if cat_name not in GROUP_MAP:
                skipped_no_map += 1
                continue

            bbox = ann["bbox"]
            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                skipped_invalid_bbox += 1
                continue

            final_class = GROUP_MAP[cat_name]
            class_id = CLASS_NAME_TO_IDX[final_class]

            entry_annotations.append({
                "category_original": cat_name,
                "class_name": final_class,
                "class_id": class_id,
                "bbox_coco": bbox,
            })

        if entry_annotations:
            dataset_entries.append({
                "image_id": img_id,
                "file_name": file_name,
                "img_path": str(img_path),
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": entry_annotations,
            })

    train_entries, temp_entries = train_test_split(
        dataset_entries,
        test_size=test_size,
        random_state=random_state,
    )

    val_entries, test_entries = train_test_split(
        temp_entries,
        test_size=val_size,
        random_state=random_state,
    )

    splits = {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }

    stats = {
        "total_images_json": len(images),
        "total_annotations_json": len(annotations),
        "usable_images": len(dataset_entries),
        "train_images": len(train_entries),
        "val_images": len(val_entries),
        "test_images": len(test_entries),
        "skipped_missing_img": skipped_missing_img,
        "skipped_no_map": skipped_no_map,
        "skipped_invalid_bbox": skipped_invalid_bbox,
    }

    return splits, stats


def save_common_dataset():
    COMMON_DIR.mkdir(parents=True, exist_ok=True)

    splits, stats = build_common_entries()

    output = {
        "stats": stats,
        "splits": splits,
    }

    out_path = COMMON_DIR / "common_dataset.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Dataset común guardado en:", out_path)
    print("Stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    save_common_dataset()