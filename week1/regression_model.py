"""Linear encoding model for task 4 in the facial-feature encoding project.

The ratings from Experiment 1 are the dependent variable and the PC scores from
task 3 are the candidate predictors. The relevant PCs are chosen with forward
selection, where each candidate model is evaluated by K-fold cross-validation
rather than by its goodness-of-fit. The final model is refitted on all data
using only the selected PCs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
RATING_DIR = ROOT / "data" / "experiment1" / DATASET_NAME
NORMALISED_DIR = ROOT / "data" / "rating_analysis" / DATASET_NAME
PCA_DIR = ROOT / "data" / "pca_analysis" / DATASET_NAME
OUTPUT_DIR = ROOT / "data" / "regression_model" / DATASET_NAME

RATING_COLUMNS = ("rating_1", "rating_2")
N_FOLDS = 10
RANDOM_SEED = 0
# Forward selection stops when the best candidate does not reduce the
# cross-validation error by at least this fraction of the current error.
MIN_RELATIVE_IMPROVEMENT = 0.001
N_PCS_TO_VISUALISE = 5


def load_ratings() -> tuple[list[str], np.ndarray, list[str]]:
    """Average each image's ratings across repetitions and participants."""
    participant_files = sorted(RATING_DIR.glob("*.csv"))
    if not participant_files:
        raise SystemExit(f"No participant CSV files found in {RATING_DIR}")

    ratings: dict[str, list[float]] = {}
    participants: list[str] = []
    for path in participant_files:
        # analyze_ratings.py writes a normalised file when a participant did
        # not use the full 1-5 scale; that file then replaces the raw ratings.
        normalised = NORMALISED_DIR / f"{path.stem}_normalized.csv"
        source = normalised if normalised.exists() else path

        with source.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        if not rows or any(not row.get(key) for row in rows for key in RATING_COLUMNS):
            print(f"{path.stem}: the experiment was not completed; the file is skipped.")
            continue

        for row in rows:
            values = [float(row[key]) for key in RATING_COLUMNS]
            ratings.setdefault(row["filename"], []).extend(values)
        participants.append(source.name)

    if not participants:
        raise SystemExit("No completed participant files were found.")

    filenames = sorted(ratings)
    mean_ratings = np.array([np.mean(ratings[name]) for name in filenames])
    return filenames, mean_ratings, participants


def load_scores(n_candidates: int) -> tuple[list[str], np.ndarray]:
    """Load the PC scores that task 3 saved for every image."""
    path = PCA_DIR / "pca_scores.csv"
    with path.open(encoding="utf-8") as file:
        rows = list(csv.reader(file))

    filenames = [row[0] for row in rows[1:]]
    scores = np.array([[float(value) for value in row[1:]] for row in rows[1:]])
    return filenames, scores[:, :n_candidates]


def fit_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the normal equations for the model y = x @ weights + intercept."""
    design = np.column_stack([x, np.ones(len(x))])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return solution[:-1], float(solution[-1])


def cross_validation_error(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Mean squared test error across the K folds for one set of predictors."""
    squared_errors = np.empty(len(y))
    for fold in range(folds.max() + 1):
        test = folds == fold
        weights, intercept = fit_least_squares(x[~test], y[~test])
        predictions = x[test] @ weights + intercept
        squared_errors[test] = (y[test] - predictions) ** 2
    return float(squared_errors.mean())


def make_folds(n_samples: int) -> np.ndarray:
    """Assign every image to one of N_FOLDS folds of near-equal size."""
    rng = np.random.default_rng(RANDOM_SEED)
    folds = np.arange(n_samples) % N_FOLDS
    return folds[rng.permutation(n_samples)]


def forward_selection(
    scores: np.ndarray, ratings: np.ndarray, folds: np.ndarray
) -> tuple[list[int], list[float], float]:
    """Add the PC that lowers the cross-validation error the most, one at a time."""
    n_candidates = scores.shape[1]
    remaining = list(range(n_candidates))
    selected: list[int] = []
    history: list[float] = []

    # The baseline model contains only the intercept, i.e. no predictors at all.
    empty = np.empty((len(ratings), 0))
    best_error = cross_validation_error(empty, ratings, folds)
    baseline_error = best_error
    print(f"Cross-validation error with the intercept only: {best_error:.4f}")

    while remaining:
        errors = [
            (cross_validation_error(scores[:, selected + [pc]], ratings, folds), pc)
            for pc in remaining
        ]
        error, pc = min(errors)

        if error > best_error * (1 - MIN_RELATIVE_IMPROVEMENT):
            break

        selected.append(pc)
        remaining.remove(pc)
        history.append(error)
        best_error = error
        print(f"  Step {len(selected):2d}: added PC{pc + 1:<3d} CV error {error:.4f}")

    if not selected:
        raise SystemExit("Forward selection did not select any PC.")
    return selected, history, baseline_error


def save_selection_figure(
    history: list[float], baseline_error: float, output_path: Path
) -> None:
    """Plot how the cross-validation error develops during forward selection."""
    steps = np.arange(len(history) + 1)
    errors = np.array([baseline_error, *history])

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(steps, errors, marker="o", color="#3973ac")
    axis.set_xlabel("Number of PCs in the model")
    axis.set_ylabel("Cross-validated mean squared error")
    axis.set_title(f"Forward selection with {N_FOLDS}-fold cross-validation")
    axis.set_xticks(steps)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_component_figure(
    mean_image: np.ndarray,
    scores: np.ndarray,
    components: np.ndarray,
    selected: list[int],
    image_shape: tuple[int, int],
    output_path: Path,
) -> None:
    """Show min-score reconstruction, mean, and max-score reconstruction per PC.

    The visualisation follows the one used in task 3, but only for the PCs that
    forward selection included in the model.
    """
    shown = selected[:N_PCS_TO_VISUALISE]
    fig, axes = plt.subplots(len(shown), 3, figsize=(9, 2.75 * len(shown)), squeeze=False)

    for row, pc_index in enumerate(shown):
        min_score = scores[:, pc_index].min()
        max_score = scores[:, pc_index].max()
        images = (
            mean_image + min_score * components[pc_index],
            mean_image,
            mean_image + max_score * components[pc_index],
        )
        titles = (
            f"PC {pc_index + 1}: minimum score ({min_score:.0f})",
            "Average image (score = 0)",
            f"Maximum score ({max_score:.0f})",
        )

        for axis, image, title in zip(axes[row], images, titles):
            axis.imshow(np.clip(image.reshape(image_shape), 0, 255), cmap="gray", vmin=0, vmax=255)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    fig.suptitle("Image variation along the PCs selected for the model", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_prediction_figure(
    ratings: np.ndarray, predictions: np.ndarray, output_path: Path
) -> None:
    """Plot the model's predicted ratings against the observed ratings."""
    fig, axis = plt.subplots(figsize=(5.5, 5.5))
    limits = (
        min(ratings.min(), predictions.min()) - 0.2,
        max(ratings.max(), predictions.max()) + 0.2,
    )
    axis.plot(limits, limits, color="#c43c39", linestyle="--", linewidth=1.25, label="y = x")
    axis.scatter(ratings, predictions, s=14, alpha=0.6, color="#3973ac")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Observed mean rating")
    axis.set_ylabel("Predicted rating")
    axis.set_title("Predicted versus observed ratings")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_predictions(
    filenames: list[str], ratings: np.ndarray, predictions: np.ndarray, output_path: Path
) -> None:
    """Save observed and predicted ratings so they can be reused in later tasks."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "observed_rating", "predicted_rating"])
        for name, observed, predicted in zip(filenames, ratings, predictions):
            writer.writerow([name, f"{observed:.4f}", f"{predicted:.4f}"])


def main() -> None:
    model = np.load(PCA_DIR / "pca_model_90pct.npz")
    mean_image = model["mean_image"].astype(np.float64)
    components = model["components"].astype(np.float64)
    image_shape = tuple(int(value) for value in model["image_shape"])
    n_candidates = int(model["n_selected"])

    rating_names, ratings, participants = load_ratings()
    score_names, scores = load_scores(n_candidates)

    if rating_names != score_names:
        missing = set(score_names) - set(rating_names)
        raise SystemExit(
            f"The rated images do not match the images in the PCA: {len(missing)} images "
            "have PC scores but no rating."
        )

    folds = make_folds(len(ratings))
    selected, history, baseline_error = forward_selection(scores, ratings, folds)

    # The final model is fitted to all the data using only the selected PCs.
    weights, intercept = fit_least_squares(scores[:, selected], ratings)
    predictions = scores[:, selected] @ weights + intercept
    residuals = ratings - predictions
    r_squared = 1 - residuals @ residuals / ((ratings - ratings.mean()) ** 2).sum()

    # The weight vector expressed in pixel space is what task 5 needs in order
    # to synthesise a face for a given rating.
    pixel_weights = weights @ components[selected]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_selection_figure(history, baseline_error, OUTPUT_DIR / "forward_selection.png")
    save_component_figure(
        mean_image,
        scores,
        components,
        selected,
        image_shape,
        OUTPUT_DIR / "selected_pcs.png",
    )
    save_prediction_figure(ratings, predictions, OUTPUT_DIR / "predicted_vs_observed.png")
    save_predictions(rating_names, ratings, predictions, OUTPUT_DIR / "predicted_ratings.csv")

    np.savez_compressed(
        OUTPUT_DIR / "regression_model.npz",
        selected_pcs=np.asarray(selected),
        weights=weights,
        intercept=np.asarray(intercept),
        pixel_weights=pixel_weights.astype(np.float32),
        mean_image=mean_image.astype(np.float32),
        image_shape=np.asarray(image_shape),
        predicted_ratings=predictions,
    )

    print(f"\nRatings from {len(participants)} participant file(s): {', '.join(participants)}")
    print(f"{len(ratings)} rated images and {n_candidates} candidate PCs.")
    print(f"Forward selection kept {len(selected)} PCs: "
          f"{', '.join(f'PC{pc + 1}' for pc in selected)}")
    print(f"Cross-validation error: {baseline_error:.4f} (intercept only) "
          f"-> {history[-1]:.4f} (final model).")
    print("Final model fitted on all data:")
    print(f"  intercept (delta): {intercept:.4f}")
    for pc, weight in zip(selected, weights):
        print(f"  PC{pc + 1}: {weight:.6g}")
    print(f"  R^2 on all data: {r_squared:.4f}")
    print(f"  Correlation between predicted and observed ratings: "
          f"{np.corrcoef(ratings, predictions)[0, 1]:.4f}")
    print(f"  Predicted ratings range from {predictions.min():.2f} to {predictions.max():.2f} "
          f"(observed: {ratings.min():.2f} to {ratings.max():.2f}).")
    print(f"Saved results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
