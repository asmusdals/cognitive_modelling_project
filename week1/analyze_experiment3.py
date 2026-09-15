"""Analysis of the adaptation experiment (task 8).

Experiment 3 adapted the observer to one endpoint of the continuum defined by
the encoding model and then asked for a rating of a near-neutral test face. The
predicted after-effect is that the test face is perceived as shifted away from
the adapting face. Adapting to the low endpoint should therefore make the same
test face look more attractive, and adapting to the high endpoint should make it
look less attractive, so

    difference = mean rating after adapting low - mean rating after adapting high

is expected to be positive for every test face. This script plots the ratings
for the three test faces separately for the two adaptation conditions and
reports, for each test face, whether the direction of the difference matches the
prediction.

Whether the difference is larger than chance is tested with a Mann-Whitney U
test, because the ratings are ordinal and the two conditions are separate sets
of trials. With only a handful of participants the direction of the difference
carries more weight than the p-value, so both are reported.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
INPUT_DIR = ROOT / "data/experiment3" / DATASET_NAME
OUTPUT_DIR = ROOT / "data/experiment3_analysis" / DATASET_NAME

CONDITIONS = ("lav", "høj")
COLOURS = {"lav": "#6aa9e0", "høj": "#e08a6a"}


def load_participants() -> list[dict]:
    """Read the Experiment 3 files and group the ratings by test face and condition."""
    participants = []

    for csv_path in sorted(INPUT_DIR.glob("*.csv")):
        with csv_path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        if not rows or any(not row.get("rating") for row in rows):
            print(f"{csv_path.stem}: forsøget er ikke fuldført; filen springes over.")
            continue

        ratings: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {condition: [] for condition in CONDITIONS}
        )
        test_ratings: dict[str, float] = {}
        for row in rows:
            if row["adapt_condition"] not in CONDITIONS:
                raise SystemExit(
                    f"{csv_path.name}: ukendt adaptationsbetingelse "
                    f"'{row['adapt_condition']}'."
                )
            ratings[row["test_filename"]][row["adapt_condition"]].append(
                float(row["rating"])
            )
            test_ratings[row["test_filename"]] = float(row["test_rating"])

        empty = [
            (name, condition)
            for name, by_condition in ratings.items()
            for condition in CONDITIONS
            if not by_condition[condition]
        ]
        if empty:
            print(
                f"{csv_path.stem}: {len(empty)} af betingelserne mangler svar; "
                "filen springes over."
            )
            continue

        participants.append(
            {
                "name": csv_path.stem,
                "test_names": sorted(test_ratings, key=lambda name: test_ratings[name]),
                "test_ratings": test_ratings,
                "ratings": {name: dict(value) for name, value in ratings.items()},
            }
        )

    return participants


def pool_participants(participants: list[dict]) -> dict:
    """Put every participant's trials into one set of six condition cells."""
    test_ratings = participants[0]["test_ratings"]
    ratings = {
        name: {
            condition: [
                value
                for entry in participants
                for value in entry["ratings"][name][condition]
            ]
            for condition in CONDITIONS
        }
        for name in test_ratings
    }
    return {
        "name": "alle",
        "test_names": participants[0]["test_names"],
        "test_ratings": test_ratings,
        "ratings": ratings,
    }


def compare_conditions(low: list[float], high: list[float]) -> dict:
    """The after-effect for one test face: does adapting low raise the rating?"""
    low_values, high_values = np.array(low), np.array(high)
    difference = float(low_values.mean() - high_values.mean())

    if len(low_values) < 2 or len(high_values) < 2 or np.std(np.r_[low_values, high_values]) == 0:
        p_value = float("nan")
    else:
        p_value = float(stats.mannwhitneyu(low_values, high_values, alternative="two-sided").pvalue)

    return {
        "n_low": len(low_values),
        "n_high": len(high_values),
        "mean_low": float(low_values.mean()),
        "mean_high": float(high_values.mean()),
        "difference": difference,
        # A difference of exactly zero is no evidence in either direction.
        "consistent": bool(difference > 0),
        "p_value": p_value,
    }


def analyse(entry: dict) -> list[dict]:
    """Compare the two adaptation conditions for each of the test faces."""
    results = []
    for name in entry["test_names"]:
        comparison = compare_conditions(
            entry["ratings"][name]["lav"], entry["ratings"][name]["høj"]
        )
        comparison["test_filename"] = name
        comparison["test_rating"] = entry["test_ratings"][name]
        results.append(comparison)
    return results


def format_p(p_value: float) -> str:
    if np.isnan(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0,001"
    return f"p = {p_value:.3f}".replace(".", ",")


def draw_grouped_boxplot(axis, entry: dict, results: list[dict], title: str) -> None:
    """Ratings of the three test faces, split by which endpoint was adapted to."""
    positions = np.arange(len(entry["test_names"]), dtype=float)

    for offset, condition in zip((-0.19, 0.19), CONDITIONS):
        data = [entry["ratings"][name][condition] for name in entry["test_names"]]
        boxes = axis.boxplot(
            data,
            positions=positions + offset,
            widths=0.32,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.4},
            flierprops={"markersize": 3, "markerfacecolor": "gray", "marker": "o"},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(COLOURS[condition])
            patch.set_alpha(0.85)

    for position, result in zip(positions, results):
        marker = "✓" if result["consistent"] else "✗"
        axis.text(
            position,
            5.42,
            f"{marker} {result['difference']:+.2f}".replace(".", ","),
            ha="center",
            fontsize=9,
            color="#2e7d32" if result["consistent"] else "#b23b3b",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{entry['test_ratings'][name]:.2f}".replace(".", ",")
            for name in entry["test_names"]
        ]
    )
    axis.set_xlim(-0.6, len(positions) - 0.4)
    axis.set_ylim(0.5, 5.7)
    axis.set_yticks(range(1, 6))
    axis.set_xlabel("Testansigt (rating det er genereret til)")
    axis.set_ylabel("Observeret rating")
    axis.set_title(title, fontsize=11)
    axis.grid(axis="y", alpha=0.3)


def add_legend(figure) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOURS[condition], alpha=0.85, edgecolor="black")
        for condition in CONDITIONS
    ]
    figure.legend(
        handles,
        ["Adapteret til lav ende", "Adapteret til høj ende"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
    )


def save_group_figure(pooled: dict, results: list[dict], n_participants: int, output_path: Path) -> None:
    """The figure task 8 asks for, with every participant's trials pooled."""
    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    draw_grouped_boxplot(
        axis,
        pooled,
        results,
        f"Alle deltagere (n = {n_participants})",
    )
    fig.suptitle("Experiment 3: efterbillede-effekt på de tre testansigter", fontsize=13)
    add_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_participant_figure(
    participants: list[dict], results: dict[str, list[dict]], output_path: Path
) -> None:
    """The same split, one panel per participant."""
    n_cols = min(3, len(participants))
    n_rows = -(-len(participants) // n_cols)  # ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows), squeeze=False
    )

    for index, entry in enumerate(participants):
        draw_grouped_boxplot(
            axes[index // n_cols][index % n_cols],
            entry,
            results[entry["name"]],
            entry["name"],
        )

    for index in range(len(participants), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    fig.suptitle("Experiment 3 pr. deltager", fontsize=14)
    add_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_summary(
    participants: list[dict],
    results: dict[str, list[dict]],
    pooled_results: list[dict],
    output_path: Path,
) -> None:
    """One row per participant and test face, plus the pooled rows."""
    fields = [
        "participant",
        "test_filename",
        "test_rating",
        "n_adapt_low",
        "n_adapt_high",
        "mean_adapt_low",
        "mean_adapt_high",
        "difference_low_minus_high",
        "direction_as_predicted",
        "mannwhitney_p",
    ]

    rows = [(entry["name"], results[entry["name"]]) for entry in participants]
    rows.append(("alle", pooled_results))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for name, entries in rows:
            for result in entries:
                writer.writerow(
                    [
                        name,
                        result["test_filename"],
                        f"{result['test_rating']:.6f}",
                        result["n_low"],
                        result["n_high"],
                        f"{result['mean_low']:.4f}",
                        f"{result['mean_high']:.4f}",
                        f"{result['difference']:+.4f}",
                        "ja" if result["consistent"] else "nej",
                        f"{result['p_value']:.6f}",
                    ]
                )


def main():
    participants = load_participants()
    if not participants:
        raise SystemExit(
            f"Der blev ikke fundet nogen brugbare CSV-filer i {INPUT_DIR}.\n"
            "Kør week1/experiment3.py for at indsamle data først."
        )

    test_names = participants[0]["test_names"]
    for entry in participants[1:]:
        if entry["test_names"] != test_names:
            raise SystemExit(
                f"{entry['name']} har set andre testansigter end "
                f"{participants[0]['name']}; deltagerne kan ikke sammenlignes."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {entry["name"]: analyse(entry) for entry in participants}
    pooled = pool_participants(participants)
    pooled_results = analyse(pooled)

    save_group_figure(
        pooled, pooled_results, len(participants), OUTPUT_DIR / "experiment3_aftereffect.png"
    )
    save_participant_figure(
        participants, results, OUTPUT_DIR / "experiment3_per_participant.png"
    )
    save_summary(participants, results, pooled_results, OUTPUT_DIR / "experiment3_summary.csv")

    print(
        f"{len(participants)} deltagere, {len(test_names)} testansigter, "
        "to adaptationsbetingelser."
    )
    print(
        "Forventet efterbillede-effekt: adaptation til den lave ende giver en "
        "højere rating end adaptation til den høje ende."
    )
    for entry in participants:
        consistent = sum(result["consistent"] for result in results[entry["name"]])
        print(f"  {entry['name']}: {consistent} af {len(test_names)} testansigter i den forventede retning")

    print("  Alle deltagere samlet:")
    for result in pooled_results:
        direction = "som forventet" if result["consistent"] else "modsat forventet"
        print(
            f"    testansigt {result['test_rating']:.2f}: "
            f"lav {result['mean_low']:.2f} mod høj {result['mean_high']:.2f}, "
            f"forskel {result['difference']:+.2f} ({direction}, {format_p(result['p_value'])})"
        )

    consistent = sum(result["consistent"] for result in pooled_results)
    print(
        f"  Samlet er {consistent} af {len(pooled_results)} testansigter i den "
        "retning, efterbillede-effekten forudsiger."
    )
    print(f"Resultaterne er gemt i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
