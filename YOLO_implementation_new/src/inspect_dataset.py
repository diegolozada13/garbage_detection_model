from pathlib import Path

DATASET_DIR = Path("raw_new/litter_detection_dataset")

print("Dataset existe:", DATASET_DIR.exists())
print("\nContenido principal:")

for p in DATASET_DIR.iterdir():
    print("-", p.name)

print("\nPrimeros archivos encontrados:")
for p in list(DATASET_DIR.rglob("*"))[:100]:
    print(p)