"""Synthetic faces for tasks 5 and 6 in the facial-feature encoding project.

Task 5 inverts the linear encoding model from task 4. The model predicts a
rating from an image, and here we go the other way: for a wanted rating a0 we
construct the unique image that carries no information other than what the
rating task is about. Following Section 2.5 in the lecture notes, that image is
the one whose low-dimensional representation is parallel to the weight vector,

    j0 = alpha * w,    alpha = (a0 - delta) / ||w||^2,

so that j0^T w + delta = a0 exactly. Everything orthogonal to w is left at the
average face, because such components do not change the predicted rating.

Task 6 checks whether the wanted ratings are actually covered by the data. The
scale factor alpha grows without bound when ||w|| -> 0, so a weak rating effect
pushes the synthetic images far outside the distribution of the input images,
where the linear model is pure extrapolation. The script therefore compares the
requested ratings with the range of ratings that the model predicts for the
training images, and generates a second set of faces inside that range.

The script also builds the stimuli for the adaptation experiment in task 8: the
two endpoints of that continuum are the adapting faces, and three faces close to
the neutral face are the test stimuli.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
MODEL_DIR = ROOT / "data" / "regression_model" / DATASET_NAME
PCA_DIR = ROOT / "data" / "pca_analysis" / DATASET_NAME
OUTPUT_DIR = ROOT / "data" / "synthetic_faces" / DATASET_NAME

# Task 5: one face per rating level used in Experiment 1 (1-5) plus the six
# half-step ratings that lie between and just outside them.
REQUESTED_RATINGS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5)
# Task 6: the replacement set uses the same number of faces, but spaced inside
# the range of ratings that the model predicts for the images it was fitted on.
N_IN_RANGE = len(REQUESTED_RATINGS)
# A requested rating counts as an extrapolation if it lies this far outside the
# predicted range, measured as a fraction of that range.
EXTRAPOLATION_TOLERANCE = 0.05
# Task 8 (Experiment 3) adapts the observer to one endpoint of the continuum and
# then measures how a near-neutral test face is perceived. The neutral face is
# the average face itself: at the rating delta the scale factor alpha is zero, so
# nothing is added to the average. The three test faces sit one step on either
# side of it, close enough that an after-effect can push the rating either way.
# The step is a compromise: one rating unit is only about eight grey levels, so a
# smaller offset makes the three test faces indistinguishable to the observer,
# while a larger one moves them away from the neutral face.
TEST_STIMULUS_OFFSET = 0.5


def load_model() -> dict[str, object]:
    """Load the encoding model that task 4 fitted and saved."""
    path = MODEL_DIR / "regression_model.npz"
    if not path.exists():
        raise SystemExit(f"{path} was not found. Run week1/regression_model.py first.")

    model = np.load(path)
    return {
        "selected_pcs": model["selected_pcs"].astype(int),
        "weights": model["weights"].astype(np.float64),
        "intercept": float(model["intercept"]),
        "pixel_weights": model["pixel_weights"].astype(np.float64),
        "mean_image": model["mean_image"].astype(np.float64),
        "image_shape": tuple(int(value) for value in model["image_shape"]),
        "predicted_ratings": model["predicted_ratings"].astype(np.float64),
    }


def load_observed_scores(selected_pcs: np.ndarray) -> np.ndarray | None:
    """Load the observed PC scores for the selected PCs, if task 3 saved them."""
    path = PCA_DIR / "pca_scores.csv"
    if not path.exists():
        return None

    with path.open(encoding="utf-8") as file:
        rows = list(csv.reader(file))
    scores = np.array([[float(value) for value in row[1:]] for row in rows[1:]])
    return scores[:, selected_pcs]


def synthesise(rating: float, model: dict[str, object]) -> tuple[np.ndarray, float, np.ndarray]:
    """Return the image, the scale factor alpha, and the PC scores for one rating.

    The image is the average face plus alpha times the weight vector expressed
    in pixel space. Because the principal components are orthonormal, adding
    alpha * (w @ components) in pixel space is the same as giving the image the
    PC scores alpha * w and then reconstructing it.
    """
    weights = model["weights"]
    alpha = (rating - model["intercept"]) / float(weights @ weights)
    image = model["mean_image"] + alpha * model["pixel_weights"]
    return image, alpha, alpha * weights


def clipped_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that fall outside the displayable range 0-255."""
    return float(np.mean((image < 0) | (image > 255)))


def out_of_range_pcs(scores: np.ndarray, observed: np.ndarray | None) -> int | None:
    """Count selected PCs whose synthetic score lies outside the observed range."""
    if observed is None:
        return None
    below = scores < observed.min(axis=0)
    above = scores > observed.max(axis=0)
    return int(np.count_nonzero(below | above))


def save_stimulus_images(
    ratings: tuple[float, ...] | np.ndarray,
    images: list[np.ndarray],
    image_shape: tuple[int, int],
    directory: Path,
) -> list[str]:
    """Write one 8-bit PNG per rating, ready to be used as stimuli in Experiment 2."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.png"):
        old.unlink()

    filenames: list[str] = []
    for rating, image in zip(ratings, images):
        pixels = np.clip(image.reshape(image_shape), 0, 255).round().astype(np.uint8)
        name = f"synthetic_rating_{float(rating):05.2f}.png"
        Image.fromarray(pixels, mode="L").save(directory / name)
        filenames.append(name)
    return filenames


def save_face_row(
    ratings: tuple[float, ...] | np.ndarray,
    images: list[np.ndarray],
    image_shape: tuple[int, int],
    title: str,
    output_path: Path,
    n_cols: int | None = None,
) -> None:
    """Show the synthetic faces on a common continuum, as in Fig. 2.6.

    By default all faces are laid out in a single row. Pass n_cols to wrap the
    continuum onto multiple rows instead (e.g. n_cols=6 puts 11 faces on two
    rows of 6 and 5), which is often easier to fit on a report page.
    """
    n_faces = len(images)
    n_cols = n_cols or n_faces
    n_rows = -(-n_faces // n_cols)  # ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.55 * n_cols, 2.6 * n_rows), squeeze=False
    )

    for index, (rating, image) in enumerate(zip(ratings, images)):
        axis = axes[index // n_cols][index % n_cols]
        axis.imshow(
            np.clip(image.reshape(image_shape), 0, 255), cmap="gray", vmin=0, vmax=255
        )
        axis.set_title(f"{float(rating):.2f}", fontsize=10)
        axis.axis("off")

    # Turn off any unused slots on the last, partially filled row.
    for index in range(n_faces, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(
    requested_ratings: tuple[float, ...],
    requested_images: list[np.ndarray],
    in_range_ratings: np.ndarray,
    in_range_images: list[np.ndarray],
    image_shape: tuple[int, int],
    output_path: Path,
) -> None:
    """Put the requested and the in-range continuum in the same figure."""
    n_faces = len(requested_images)
    fig, axes = plt.subplots(2, n_faces, figsize=(1.55 * n_faces, 5.4), squeeze=False)

    rows = (
        (requested_ratings, requested_images, "Requested (task 5)"),
        (in_range_ratings, in_range_images, "Inside range (task 6)"),
    )
    for row, (ratings, images, label) in enumerate(rows):
        for column, (axis, rating, image) in enumerate(zip(axes[row], ratings, images)):
            axis.imshow(
                np.clip(image.reshape(image_shape), 0, 255), cmap="gray", vmin=0, vmax=255
            )
            axis.set_title(f"{float(rating):.2f}", fontsize=10)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.14,
                    0.5,
                    label,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                )

    fig.suptitle(
        "Synthetic faces before and after restricting the rating range", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_range_figure(
    predicted: np.ndarray,
    requested_ratings: tuple[float, ...],
    in_range_ratings: np.ndarray,
    output_path: Path,
) -> None:
    """Show the predicted ratings of the training images against both stimulus sets."""
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.hist(
        predicted,
        bins=30,
        color="#3973ac",
        alpha=0.75,
        label="Model prediction for the rated images",
    )
    axis.axvspan(
        predicted.min(),
        predicted.max(),
        color="#3973ac",
        alpha=0.12,
        label=f"Predicted range ({predicted.min():.2f} to {predicted.max():.2f})",
    )

    top = axis.get_ylim()[1]
    axis.vlines(
        requested_ratings,
        0,
        top,
        color="#c43c39",
        linestyle="--",
        linewidth=1.25,
        label="Requested ratings (task 5)",
    )
    axis.vlines(
        in_range_ratings,
        0,
        top * 0.55,
        color="#2e8b57",
        linewidth=1.75,
        label="New stimuli inside the range (task 6)",
    )
    axis.set_ylim(0, top)
    axis.set_xlabel("Rating")
    axis.set_ylabel("Number of images")
    axis.set_title("Requested ratings versus the ratings the model actually predicts")
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_experiment3_figure(
    adapt_ratings: np.ndarray,
    adapt_images: list[np.ndarray],
    test_ratings: np.ndarray,
    test_images: list[np.ndarray],
    image_shape: tuple[int, int],
    output_path: Path,
) -> None:
    """Show the adapting and test stimuli of the adaptation experiment together."""
    panels = (
        [(adapt_ratings[0], adapt_images[0], "Adapter (lav)")]
        + [(rating, image, "Test") for rating, image in zip(test_ratings, test_images)]
        + [(adapt_ratings[-1], adapt_images[-1], "Adapter (høj)")]
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(1.7 * len(panels), 2.9))
    for axis, (rating, image, label) in zip(axes, panels):
        axis.imshow(
            np.clip(image.reshape(image_shape), 0, 255), cmap="gray", vmin=0, vmax=255
        )
        axis.set_title(f"{label}\n{float(rating):.2f}", fontsize=10)
        axis.axis("off")

    fig.suptitle("Stimuli til Experiment 3: adaptere og testansigter", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save one row per synthetic image, so Experiment 2 can join ratings to predictions."""
    fieldnames = [
        "set",
        "filename",
        "target_rating",
        "target_rating_exact",
        "alpha",
        "clipped_pixel_fraction",
        "pcs_outside_observed_range",
        "n_selected_pcs",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_set(
    ratings: tuple[float, ...] | np.ndarray,
    model: dict[str, object],
    observed_scores: np.ndarray | None,
    set_name: str,
    stimulus_dir: Path,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Synthesise one continuum of faces and collect its diagnostics."""
    images: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for rating in ratings:
        image, alpha, scores = synthesise(float(rating), model)
        images.append(image)
        rows.append(
            {
                "set": set_name,
                "filename": "",
                "target_rating": f"{float(rating):.2f}",
                # The filename only carries two decimals; task 7 correlates the
                # observed ratings against the exact value used to build the image.
                "target_rating_exact": f"{float(rating):.6f}",
                "alpha": f"{alpha:.6g}",
                "clipped_pixel_fraction": f"{clipped_fraction(image):.4f}",
                "pcs_outside_observed_range": out_of_range_pcs(scores, observed_scores),
                "n_selected_pcs": len(model["selected_pcs"]),
            }
        )

    filenames = save_stimulus_images(ratings, images, model["image_shape"], stimulus_dir)
    for row, name in zip(rows, filenames):
        row["filename"] = name
    return images, rows


def main() -> None:
    model = load_model()
    observed_scores = load_observed_scores(model["selected_pcs"])
    predicted = model["predicted_ratings"]
    weights = model["weights"]
    weight_norm = float(np.linalg.norm(weights))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Task 5: the faces that were asked for -----------------------------
    requested_images, requested_rows = build_set(
        REQUESTED_RATINGS,
        model,
        observed_scores,
        "requested",
        OUTPUT_DIR / "stimuli_requested",
    )
    save_face_row(
        REQUESTED_RATINGS,
        requested_images,
        model["image_shape"],
        "Synthetic faces in rating range 0.5 to 5.5 ",
        OUTPUT_DIR / "synthetic_faces_requested.png",
        n_cols=6,
    )

    # --- Task 6: is that range covered by the data? ------------------------
    lower, upper = float(predicted.min()), float(predicted.max())
    tolerance = EXTRAPOLATION_TOLERANCE * (upper - lower)
    outside = [
        rating
        for rating in REQUESTED_RATINGS
        if rating < lower - tolerance or rating > upper + tolerance
    ]
    in_range_ratings = np.linspace(lower, upper, N_IN_RANGE)

    in_range_images, in_range_rows = build_set(
        in_range_ratings,
        model,
        observed_scores,
        "in_range",
        OUTPUT_DIR / "stimuli_in_range",
    )
    save_face_row(
        in_range_ratings,
        in_range_images,
        model["image_shape"],
        "Synthetic faces inside the range of predicted ratings",
        OUTPUT_DIR / "synthetic_faces_in_range.png",
        n_cols=6,
    )
    save_comparison_figure(
        REQUESTED_RATINGS,
        requested_images,
        in_range_ratings,
        in_range_images,
        model["image_shape"],
        OUTPUT_DIR / "synthetic_faces_comparison.png",
    )
    save_range_figure(
        predicted,
        REQUESTED_RATINGS,
        in_range_ratings,
        OUTPUT_DIR / "predicted_rating_range.png",
    )

    # --- Task 8: stimuli for the adaptation experiment ---------------------
    # The adapters are the two endpoints of the continuum from task 6, so they
    # are the strongest faces the model supports without extrapolating.
    adapt_ratings = np.array([in_range_ratings[0], in_range_ratings[-1]])
    test_ratings = model["intercept"] + TEST_STIMULUS_OFFSET * np.array([-1.0, 0.0, 1.0])

    adapt_images, adapt_rows = build_set(
        adapt_ratings,
        model,
        observed_scores,
        "exp3_adapt",
        OUTPUT_DIR / "stimuli_experiment3_adapt",
    )
    test_images, test_rows = build_set(
        test_ratings,
        model,
        observed_scores,
        "exp3_test",
        OUTPUT_DIR / "stimuli_experiment3_test",
    )
    save_experiment3_figure(
        adapt_ratings,
        adapt_images,
        test_ratings,
        test_images,
        model["image_shape"],
        OUTPUT_DIR / "experiment3_stimuli.png",
    )

    save_manifest(
        requested_rows + in_range_rows + adapt_rows + test_rows,
        OUTPUT_DIR / "synthetic_faces.csv",
    )

    np.savez_compressed(
        OUTPUT_DIR / "synthetic_faces.npz",
        requested_ratings=np.asarray(REQUESTED_RATINGS),
        requested_images=np.stack(requested_images).astype(np.float32),
        in_range_ratings=in_range_ratings,
        in_range_images=np.stack(in_range_images).astype(np.float32),
        exp3_adapt_ratings=adapt_ratings,
        exp3_adapt_images=np.stack(adapt_images).astype(np.float32),
        exp3_test_ratings=test_ratings,
        exp3_test_images=np.stack(test_images).astype(np.float32),
        image_shape=np.asarray(model["image_shape"]),
        predicted_rating_min=np.asarray(lower),
        predicted_rating_max=np.asarray(upper),
    )

    # --- Report-ready summary ----------------------------------------------
    n_pixels = model["mean_image"].size
    print(
        f"Encoding model: {len(model['selected_pcs'])} PCs, "
        f"intercept (delta) = {model['intercept']:.4f}"
    )
    print(f"Weight vector: ||w|| = {weight_norm:.6g}, ||w||^2 = {weight_norm ** 2:.6g}")
    print(
        "Distance from the average face per rating step: "
        f"1/||w|| = {1 / weight_norm:.1f} in L2 norm, i.e. "
        f"{1 / weight_norm / np.sqrt(n_pixels):.2f} grey levels per pixel (RMS)."
    )

    print("\nTask 5 - requested ratings")
    for row in requested_rows:
        print(
            f"  a0 = {row['target_rating']:>5}: alpha = {float(row['alpha']):+.3e}, "
            f"{float(row['clipped_pixel_fraction']) * 100:5.1f}% of pixels clipped, "
            f"{row['pcs_outside_observed_range']}/{row['n_selected_pcs']} PC scores "
            "outside the observed range"
        )

    print("\nTask 6 - the range the model actually predicts")
    print(f"  Minimum predicted rating: {lower:.4f}")
    print(f"  Maximum predicted rating: {upper:.4f}")
    print(
        f"  Requested range: {min(REQUESTED_RATINGS):.2f} to {max(REQUESTED_RATINGS):.2f}"
    )
    if outside:
        print(
            f"  {len(outside)} of {len(REQUESTED_RATINGS)} requested ratings lie outside "
            "the predicted range and are extrapolations: "
            + ", ".join(f"{value:.2f}" for value in outside)
        )
        print(
            f"  A new set of {N_IN_RANGE} faces was generated from {lower:.2f} to "
            f"{upper:.2f} and should be used in Experiment 2."
        )
    else:
        print(
            "  All requested ratings lie inside the predicted range; no new set is needed."
        )

    requested_clipped = np.mean(
        [float(row["clipped_pixel_fraction"]) for row in requested_rows]
    )
    in_range_clipped = np.mean(
        [float(row["clipped_pixel_fraction"]) for row in in_range_rows]
    )
    print(
        f"  Mean clipped pixels: {requested_clipped * 100:.1f}% (requested) versus "
        f"{in_range_clipped * 100:.1f}% (in range)."
    )
    if observed_scores is not None:
        requested_outside = np.mean(
            [row["pcs_outside_observed_range"] for row in requested_rows]
        )
        in_range_outside = np.mean(
            [row["pcs_outside_observed_range"] for row in in_range_rows]
        )
        print(
            "  Mean number of selected PCs with a score outside the observed range: "
            f"{requested_outside:.1f} (requested) versus {in_range_outside:.1f} "
            f"(in range), out of {len(model['selected_pcs'])} PCs."
        )

    print("\nTask 8 - stimuli for the adaptation experiment")
    print(
        "  Adapters at the endpoints of the continuum: "
        + " and ".join(f"{value:.2f}" for value in adapt_ratings)
    )
    print(
        f"  Test faces around the neutral face (delta = {model['intercept']:.2f}): "
        + ", ".join(f"{value:.2f}" for value in test_ratings)
    )
    # The test faces are useless if the observer cannot tell them apart, so
    # report how far apart they actually are in the image.
    test_stack = np.stack([np.clip(image, 0, 255) for image in test_images])
    neighbour_rms = [
        float(np.sqrt(np.mean((test_stack[index + 1] - test_stack[index]) ** 2)))
        for index in range(len(test_stack) - 1)
    ]
    adapt_rms = float(
        np.sqrt(np.mean((np.clip(adapt_images[-1], 0, 255) - np.clip(adapt_images[0], 0, 255)) ** 2))
    )
    print(
        "  Distance between neighbouring test faces: "
        + ", ".join(f"{value:.1f}" for value in neighbour_rms)
        + f" grey levels (RMS), against {adapt_rms:.1f} between the two adapters."
    )
    print(f"\nSaved results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
