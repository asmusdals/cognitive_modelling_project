from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
SOURCE_DIR = ROOT / "data/raw" / DATASET_NAME
OUTPUT_DIR = ROOT / "data/processed" / DATASET_NAME


def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Kildemappen findes ikke: {SOURCE_DIR}")

    # Every JPG in this directory has already passed the manual selection.
    files = sorted(SOURCE_DIR.glob("*.jpg"))
    if not files:
        raise SystemExit(f"Der blev ikke fundet nogen JPG-billeder i {SOURCE_DIR}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        with Image.open(path) as image:
            image.convert("L").save(OUTPUT_DIR / path.name)

    print(f"Konverterede {len(files)} billeder til gråtoner i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
