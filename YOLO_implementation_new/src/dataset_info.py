from pathlib import Path
from collections import Counter

LABELS_DIR = Path("raw_new/litter_detection_dataset/data/labels/train")

counter = Counter()

for txt in LABELS_DIR.glob("*.txt"):
    for line in txt.read_text().splitlines():
        if not line.strip():
            continue
        cls = int(line.split()[0])
        counter[cls] += 1

names = {
    0: "Metal",
    1: "Cardboard",
    2: "Glass",
    3: "Plastic",
}

for cls, count in sorted(counter.items()):
    print(names[cls], count)