"""Analysis of Experiment 2 (task 7).

Experiment 2 showed the synthetic faces from task 6 to the participants, who
rated them on the same 1-5 scale as in Experiment 1. Every face was constructed
to have a known predicted rating, so the model is validated by asking whether
the ratings the participants actually gave follow those predictions.

Task 7 asks for a box plot of the ratings as a function of the predicted rating
for each participant, and for Spearman's rank correlation coefficient rho. Rho
is used rather than Pearson's r because the ratings are ordinal: a participant
who orders the faces correctly but uses the scale unevenly should still count as
agreeing with the model. For the same reason the raw ratings are used
throughout. Rho only depends on the ranks, so the min-max normalisation from
Experiment 1 would leave every value in this script unchanged.

Rho is expected to be close to 1. How close it can realistically get is limited
by how consistently a participant rates the same face twice, so the split-half
reliability of each participant is reported next to their rho.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
INPUT_DIR = ROOT / "data/experiment2" / DATASET_NAME
MANIFEST_FILE = ROOT / "data/synthetic_faces" / DATASET_NAME / "synthetic_faces.csv"
OUTPUT_DIR = ROOT / "data/experiment2_analysis" / DATASET_NAME
STIMULUS_SET = "in_range"


def load_targets() -> dict[str, float]:
    """Map each stimulus filename to the rating the model predicts for it."""
    if not MANIFEST_FILE.exists():
        raise SystemExit(
            f"{MANIFEST_FILE} blev ikke fundet. Kør week1/synthetic_faces.py først."
        )

    with MANIFEST_FILE.open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["set"] == STIMULUS_SET]

    return {row["filename"]: float(row["target_rating_exact"]) for row in rows}


def load_participants(targets: dict[str, float]) -> list[dict]:
    """Read the completed Experiment 2 files, one entry per participant."""
    participants = []

    for csv_path in sorted(INPUT_DIR.glob("*.csv")):
        with csv_path.open(encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rating_columns = [name for name in reader.fieldnames if name.startswith("rating_")]
            rows = list(reader)

        if not rows or any(not row.get(name) for row in rows for name in rating_columns):
            print(f"{csv_path.stem}: forsøget er ikke fuldført; filen springes over.")
            continue

        # The manifest is authoritative, but a file recorded with another
        # stimulus set still carries its own predicted rating and can be analysed.
        rows.sort(key=lambda row: targets.get(row["filename"], float(row["target_rating"])))
        unknown = [row["filename"] for row in rows if row["filename"] not in targets]
        if unknown:
            print(f"{csv_path.stem}: {len(unknown)} billeder findes ikke i manifestet.")

        participants.append(
            {
                "name": csv_path.stem,
                "filenames": [row["filename"] for row in rows],
                "targets": np.array(
                    [
                        targets.get(row["filename"], float(row["target_rating"]))
                        for row in rows
                    ]
                ),
                "ratings": np.array(
                    [[float(row[name]) for name in rating_columns] for row in rows]
                ),
            }
        )

    return participants


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman's rho and its p-value, without failing on a constant variable."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def split_half_reliability(ratings: np.ndarray) -> float:
    """How consistently the participant rated the same face, on a 0-1 scale.

    The odd and even repetitions are averaged separately and correlated, and the
    result is corrected back up to the full number of repetitions with the
    Spearman-Brown formula. It is the ceiling that rho can realistically reach:
    a participant who cannot reproduce their own ratings cannot agree with the
    model either.
    """
    if ratings.shape[1] < 2:
        return float("nan")
    first = ratings[:, 0::2].mean(axis=1)
    second = ratings[:, 1::2].mean(axis=1)
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    r = float(np.corrcoef(first, second)[0, 1])
    if r <= -1:
        return float("nan")
    return 2 * r / (1 + r)


def analyse_participant(entry: dict) -> dict:
    """Rank-correlate one participant's ratings with the predicted ratings."""
    ratings = entry["ratings"]
    targets = entry["targets"]

    # Every presentation is one observation, so each predicted rating is
    # repeated once per presentation of that face.
    trial_targets = np.repeat(targets, ratings.shape[1])
    rho, p_value = spearman(trial_targets, ratings.reshape(-1))
    # The same correlation on the per-face averages, where the within-face
    # noise has been averaged out, is normally noticeably higher.
    rho_means, p_means = spearman(targets, ratings.mean(axis=1))
    pearson = (
        float(np.corrcoef(trial_targets, ratings.reshape(-1))[0, 1])
        if np.std(ratings) > 0
        else float("nan")
    )

    return {
        "name": entry["name"],
        "filenames": entry["filenames"],
        "targets": targets,
        "ratings": ratings,
        "used": np.unique(ratings).tolist(),
        "means": ratings.mean(axis=1),
        "medians": np.median(ratings, axis=1),
        "rho": rho,
        "p_value": p_value,
        "rho_means": rho_means,
        "p_means": p_means,
        "pearson": pearson,
        "split_half": split_half_reliability(ratings),
    }


def format_p(p_value: float) -> str:
    if np.isnan(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0,001"
    return f"p = {p_value:.3f}".replace(".", ",")


def draw_boxplot(axis, targets: np.ndarray, ratings: np.ndarray, title: str) -> None:
    """Box plot of the ratings given to each face, placed at its predicted rating."""
    spacing = float(np.min(np.diff(targets))) if len(targets) > 1 else 1.0

    axis.plot(
        [targets.min() - spacing, targets.max() + spacing],
        [targets.min() - spacing, targets.max() + spacing],
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=1,
    )
    axis.boxplot(
        [ratings[index] for index in range(len(targets))],
        positions=targets,
        widths=spacing * 0.6,
        medianprops={"color": "crimson", "linewidth": 1.6},
        flierprops={"markersize": 3, "markerfacecolor": "gray", "marker": "o"},
        zorder=2,
    )

    axis.set_xlim(targets.min() - spacing, targets.max() + spacing)
    axis.set_ylim(0.5, 5.5)
    axis.set_yticks(range(1, 6))
    axis.set_xticks(targets)
    axis.set_xticklabels([f"{value:.2f}" for value in targets], rotation=90, fontsize=8)
    axis.set_xlabel("Forudsagt rating")
    axis.set_ylabel("Observeret rating")
    axis.set_title(title, fontsize=11)
    axis.grid(axis="y", alpha=0.3)


def save_participant_boxplots(results: list[dict], output_path: Path) -> None:
    """The figure task 7 asks for: one box plot per test participant."""
    n_cols = min(3, len(results))
    n_rows = -(-len(results) // n_cols)  # ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 4.4 * n_rows), squeeze=False
    )

    for index, result in enumerate(results):
        title = (
            f"{result['name']}: ρ = {result['rho']:.2f}".replace(".", ",")
            + f", {format_p(result['p_value'])}"
        )
        draw_boxplot(axes[index // n_cols][index % n_cols], result["targets"], result["ratings"], title)

    for index in range(len(results), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    fig.suptitle("Experiment 2: ratings som funktion af forudsagt rating", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_group_boxplot(results: list[dict], pooled: dict, output_path: Path) -> None:
    """The same box plot with every participant's trials pooled together."""
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    title = (
        f"Alle deltagere: ρ = {pooled['rho']:.2f}".replace(".", ",")
        + f", {format_p(pooled['p_value'])}"
        + f", {len(results)} deltagere"
    )
    draw_boxplot(axis, pooled["targets"], pooled["ratings"], title)
    fig.suptitle("Experiment 2: ratings som funktion af forudsagt rating", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def pool_participants(results: list[dict]) -> dict:
    """Put every participant's trials for a given face into one distribution."""
    targets = results[0]["targets"]
    ratings = np.hstack([result["ratings"] for result in results])
    trial_targets = np.repeat(targets, ratings.shape[1])
    rho, p_value = spearman(trial_targets, ratings.reshape(-1))
    rho_means, p_means = spearman(targets, ratings.mean(axis=1))

    return {
        "name": "alle",
        "targets": targets,
        "ratings": ratings,
        "means": ratings.mean(axis=1),
        "medians": np.median(ratings, axis=1),
        "rho": rho,
        "p_value": p_value,
        "rho_means": rho_means,
        "p_means": p_means,
        "pearson": float(np.corrcoef(trial_targets, ratings.reshape(-1))[0, 1]),
    }


def save_stimulus_means(results: list[dict], pooled: dict, output_path: Path) -> None:
    """One row per synthetic face, with every participant's average rating."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["filename", "predicted_rating"]
            + [f"{result['name']}_mean" for result in results]
            + ["all_mean", "all_median", "all_std"]
        )
        for index, filename in enumerate(results[0]["filenames"]):
            writer.writerow(
                [filename, f"{pooled['targets'][index]:.6f}"]
                + [f"{result['means'][index]:.4f}" for result in results]
                + [
                    f"{pooled['means'][index]:.4f}",
                    f"{pooled['medians'][index]:.2f}",
                    f"{pooled['ratings'][index].std(ddof=1):.4f}",
                ]
            )


def save_summary(results: list[dict], pooled: dict, output_path: Path) -> None:
    """Spearman's rho per participant, which is what task 7 asks to report."""
    fields = [
        "participant",
        "n_stimuli",
        "n_repeats",
        "used_ratings",
        "spearman_rho",
        "p_value",
        "spearman_rho_face_means",
        "pearson_r",
        "split_half_reliability",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for result in results:
            writer.writerow(
                [
                    result["name"],
                    len(result["targets"]),
                    result["ratings"].shape[1],
                    " ".join(f"{value:g}" for value in result["used"]),
                    f"{result['rho']:.4f}",
                    f"{result['p_value']:.6f}",
                    f"{result['rho_means']:.4f}",
                    f"{result['pearson']:.4f}",
                    f"{result['split_half']:.4f}",
                ]
            )
        writer.writerow(
            [
                pooled["name"],
                len(pooled["targets"]),
                pooled["ratings"].shape[1],
                "",
                f"{pooled['rho']:.4f}",
                f"{pooled['p_value']:.6f}",
                f"{pooled['rho_means']:.4f}",
                f"{pooled['pearson']:.4f}",
                "",
            ]
        )


def main():
    targets = load_targets()
    participants = load_participants(targets)
    if not participants:
        raise SystemExit(
            f"Der blev ikke fundet nogen fuldførte CSV-filer i {INPUT_DIR}.\n"
            "Kør week1/experiment2.py for at indsamle data først."
        )

    stimuli = participants[0]["filenames"]
    for entry in participants[1:]:
        if entry["filenames"] != stimuli:
            raise SystemExit(
                f"{entry['name']} har set andre billeder end {participants[0]['name']}; "
                "deltagerne kan ikke sammenlignes."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [analyse_participant(entry) for entry in participants]
    pooled = pool_participants(results)

    save_participant_boxplots(results, OUTPUT_DIR / "experiment2_boxplots.png")
    save_group_boxplot(results, pooled, OUTPUT_DIR / "experiment2_group_boxplot.png")
    save_stimulus_means(results, pooled, OUTPUT_DIR / "experiment2_stimulus_means.csv")
    save_summary(results, pooled, OUTPUT_DIR / "experiment2_summary.csv")

    print(
        f"{len(results)} deltagere, {len(pooled['targets'])} syntetiske ansigter, "
        f"{results[0]['ratings'].shape[1]} gentagelser pr. ansigt."
    )
    for result in results:
        print(
            f"  {result['name']}: rho = {result['rho']:.3f} ({format_p(result['p_value'])}), "
            f"rho på ansigtsgennemsnit = {result['rho_means']:.3f}, "
            f"split-half = {result['split_half']:.3f}"
        )
    print(
        f"  Alle deltagere: rho = {pooled['rho']:.3f} ({format_p(pooled['p_value'])}), "
        f"rho på ansigtsgennemsnit = {pooled['rho_means']:.3f}"
    )
    print(f"Resultaterne er gemt i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
