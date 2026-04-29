import json
from src.config import ANNOTATIONS_JSON, IMAGES_DIR

with open(ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

missing = []

for img in coco["images"]:
    file_name = img["file_name"]
    img_path = IMAGES_DIR / file_name

    if not img_path.exists():
        missing.append(file_name)

print("Total imágenes en JSON:", len(coco["images"]))
print("Imágenes no encontradas:", len(missing))
print("Primeras 30 missing:")
for m in missing[:30]:
    print(m)

print("\nArchivos reales en raw/images:", len(list(IMAGES_DIR.rglob("*.*"))))
print("Subcarpetas en raw/images:")
for p in sorted(IMAGES_DIR.iterdir()):
    if p.is_dir():
        print(p.name, "->", len(list(p.glob('*.*'))))