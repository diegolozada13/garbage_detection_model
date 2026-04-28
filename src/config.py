from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
ANNOTATIONS_JSON = RAW_DIR / "annotations.json"

OUTPUTS_DIR = PROJECT_DIR / "outputs"
COMMON_DIR = OUTPUTS_DIR / "common"
YOLO_DIR = OUTPUTS_DIR / "yolo"
RUNS_DIR = OUTPUTS_DIR / "runs"

FINAL_CLASSES = ["amarillo", "azul", "verde", "marron", "gris"]

CLASS_NAME_TO_IDX = {
    "amarillo": 0,
    "azul": 1,
    "verde": 2,
    "marron": 3,
    "gris": 4,
}

GROUP_MAP = {
    "Aluminium foil": "amarillo",
    "Aluminium blister pack": "amarillo",
    "Carded blister pack": "amarillo",
    "Other plastic bottle": "amarillo",
    "Clear plastic bottle": "amarillo",
    "Plastic bottle cap": "amarillo",
    "Metal bottle cap": "amarillo",
    "Food Can": "amarillo",
    "Aerosol": "amarillo",
    "Drink can": "amarillo",
    "Plastic lid": "amarillo",
    "Metal lid": "amarillo",
    "Other plastic": "amarillo",
    "Plastic film": "amarillo",
    "Six pack rings": "amarillo",
    "Garbage bag": "amarillo",
    "Other plastic wrapper": "amarillo",
    "Single-use carrier bag": "amarillo",
    "Polypropylene bag": "amarillo",
    "Crisp packet": "gris",
    "Spread tub": "amarillo",
    "Tupperware": "amarillo",
    "Disposable food container": "amarillo",
    "Foam food container": "gris",
    "Other plastic container": "amarillo",
    "Plastic glooves": "amarillo",
    "Plastic utensils": "amarillo",
    "Pop tab": "amarillo",
    "Squeezable tube": "amarillo",
    "Plastic straw": "amarillo",

    "Toilet tube": "azul",
    "Other carton": "azul",
    "Egg carton": "azul",
    "Drink carton": "azul",
    "Corrugated carton": "azul",
    "Meal carton": "azul",
    "Pizza box": "azul",
    "Paper cup": "azul",
    "Magazine paper": "azul",
    "Wrapping paper": "azul",
    "Normal paper": "azul",
    "Paper bag": "azul",
    "Paper straw": "azul",

    "Glass bottle": "verde",
    "Broken glass": "verde",
    "Glass cup": "verde",
    "Glass jar": "verde",

    "Food waste": "marron",

    "Battery": "gris",
    "Foam cup": "gris",
    "Tissues": "gris",
    "Plastified paper bag": "gris",
    "Rope & strings": "gris",
    "Scrap metal": "gris",
    "Shoe": "gris",
    "Styrofoam piece": "gris",
    "Unlabeled litter": "gris",
    "Cigarette": "gris",
}