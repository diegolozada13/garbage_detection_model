from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_DIR / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"

DATASET_YAML = (
    PROJECT_DIR
    / "raw_new"
    / "litter_detection_dataset"
    / "data"
    / "data_local.yaml"
)