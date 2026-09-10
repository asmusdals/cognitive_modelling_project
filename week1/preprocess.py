from pathlib import Path

from PIL import Image


root = Path(__file__).resolve().parents[1]
source = root / "data/raw/26yearmale"
output = root / "data/processed/26yearmale"
output.mkdir(parents=True, exist_ok=True)

files = list(source.glob("26_0_0_*.jpg"))
for path in files:
    with Image.open(path) as image:
        image.convert("L").save(output / path.name)

print(f"Converted {len(files)} images to grayscale in {output}")
