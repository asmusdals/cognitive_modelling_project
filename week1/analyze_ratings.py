import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
INPUT_DIR = ROOT / "data/experiment1" / DATASET_NAME
OUTPUT_DIR = ROOT / "data/rating_analysis" / DATASET_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for csv_path in sorted(INPUT_DIR.glob("*.csv")):
    with csv_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows or any(not row.get(key) for row in rows for key in ("rating_1", "rating_2")):
        print(f"{csv_path.stem}: forsøget er ikke fuldført; filen springes over.")
        continue

    ratings = [float(row[key]) for row in rows for key in ("rating_1", "rating_2")]
    used = sorted(set(ratings))

    plt.figure(figsize=(6, 4))
    plt.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor="black")
    plt.xticks(range(1, 6))
    plt.xlabel("Rating")
    plt.ylabel("Antal")
    plt.title(f"Ratings – {csv_path.stem}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{csv_path.stem}_histogram.png", dpi=200)
    plt.close()

    print(f"{csv_path.stem}: brugte ratings {used}")
    if used != [1, 2, 3, 4, 5]:
        low, high = min(ratings), max(ratings)
        if low == high:
            print("  Kan ikke min-max-normaliseres: alle ratings er ens.")
            continue

        for row in rows:
            for key in ("rating_1", "rating_2"):
                value = float(row[key])
                row[key] = round(1 + 4 * (value - low) / (high - low), 3)

        output_csv = OUTPUT_DIR / f"{csv_path.stem}_normalized.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["filename", "rating_1", "rating_2"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Normaliseret til 1–5 og gemt som {output_csv.name}")
    else:
        print("  Hele skalaen er brugt; normalisering er ikke nødvendig.")
