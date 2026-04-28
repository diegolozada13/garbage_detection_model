import json
import math
from collections import Counter, defaultdict

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


def get_dominant_class(entry_annotations):
    counter = Counter(ann["class_name"] for ann in entry_annotations)
    return counter.most_common(1)[0][0]


def count_instances_by_class(entries):
    counter = Counter()

    for entry in entries:
        for ann in entry["annotations"]:
            counter[ann["class_name"]] += 1

    return counter


def count_images_by_dominant_class(entries):
    return Counter(entry["dominant_class"] for entry in entries)


def oversample_minority_classes(entries, threshold_ratio=0.4, max_factor=4):
    """
    Duplica imágenes del train que contienen clases minoritarias.

    threshold_ratio:
        Una clase se considera minoritaria si tiene menos del X% de instancias
        respecto a la clase mayoritaria.

    max_factor:
        Límite máximo de duplicación para evitar sobreajuste extremo.
    """

    class_counts = count_instances_by_class(entries)

    if not class_counts:
        return entries, {}

    max_count = max(class_counts.values())
    target_classes = {}

    for cls, count in class_counts.items():
        ratio = count / max_count

        if ratio < threshold_ratio:
            desired_min_count = threshold_ratio * max_count
            factor = math.ceil(desired_min_count / count)
            factor = min(factor, max_factor)

            if factor > 1:
                target_classes[cls] = factor

    extra_entries = []

    for entry in entries:
        entry_classes = {ann["class_name"] for ann in entry["annotations"]}

        factor = 1
        for cls in entry_classes:
            if cls in target_classes:
                factor = max(factor, target_classes[cls])

        if factor > 1:
            extra_entries.extend([entry] * (factor - 1))

    oversampled_entries = entries + extra_entries

    return oversampled_entries, target_classes


def build_common_entries(
    test_size=0.2,
    val_size=0.5,
    random_state=42,
    use_stratified_split=True,
    use_oversampling=True,
    oversampling_threshold_ratio=0.25,
    oversampling_max_factor=3,
):
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

            entry_annotations.append(
                {
                    "category_original": cat_name,
                    "class_name": final_class,
                    "class_id": class_id,
                    "bbox_coco": bbox,
                }
            )

        if entry_annotations:
            dominant_class = get_dominant_class(entry_annotations)

            dataset_entries.append(
                {
                    "image_id": img_id,
                    "file_name": file_name,
                    "img_path": str(img_path),
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "annotations": entry_annotations,
                    "dominant_class": dominant_class,
                }
            )

    # Split train / val / test
    if use_stratified_split:
        stratify_labels = [e["dominant_class"] for e in dataset_entries]

        train_entries, temp_entries = train_test_split(
            dataset_entries,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )

        temp_labels = [e["dominant_class"] for e in temp_entries]

        val_entries, test_entries = train_test_split(
            temp_entries,
            test_size=val_size,
            random_state=random_state,
            stratify=temp_labels,
        )
    else:
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

    train_instances_before = count_instances_by_class(train_entries)
    val_instances = count_instances_by_class(val_entries)
    test_instances = count_instances_by_class(test_entries)

    oversampled_classes = {}

    if use_oversampling:
        train_entries, oversampled_classes = oversample_minority_classes(
            train_entries,
            threshold_ratio=oversampling_threshold_ratio,
            max_factor=oversampling_max_factor,
        )

    train_instances_after = count_instances_by_class(train_entries)

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
        "use_stratified_split": use_stratified_split,
        "use_oversampling": use_oversampling,
        "oversampling_threshold_ratio": oversampling_threshold_ratio,
        "oversampling_max_factor": oversampling_max_factor,
        "oversampled_classes": oversampled_classes,
        "train_instances_before_oversampling": dict(train_instances_before),
        "train_instances_after_oversampling": dict(train_instances_after),
        "val_instances": dict(val_instances),
        "test_instances": dict(test_instances),
        "train_dominant_classes": dict(count_images_by_dominant_class(train_entries)),
        "val_dominant_classes": dict(count_images_by_dominant_class(val_entries)),
        "test_dominant_classes": dict(count_images_by_dominant_class(test_entries)),
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
    print("\nStats:")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    save_common_dataset()