"""PCA analysis for task 3 in the facial-feature encoding project.

The script treats each grayscale image as one observation and each pixel as one
feature. It subtracts the mean image, but deliberately does not divide pixels by
their standard deviations. PCA is then computed with an economy-size SVD.
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
INPUT_DIR = ROOT / "data" / "processed" / DATASET_NAME
OUTPUT_DIR = ROOT / "data" / "pca_analysis" / DATASET_NAME

# An unsupervised cutoff used to define the candidate predictors for task 4.
VARIANCE_THRESHOLD = 0.90
N_PCS_TO_VISUALISE = 5


def load_images(input_dir: Path) -> tuple[list[Path], np.ndarray, tuple[int, int]]:
    """Load equally sized grayscale images into an (images x pixels) matrix."""
    paths = sorted(input_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No JPG images found in {input_dir}")

    flattened: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float64)
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"All images must have the same shape; {path.name} has {array.shape}, "
                f"expected {expected_shape}."
            )
        flattened.append(array.ravel())

    assert expected_shape is not None
    return paths, np.stack(flattened), expected_shape


def compute_pca(
    images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean image, PC scores, components, and explained-variance ratios."""
    mean_image = images.mean(axis=0)
    centered = images - mean_image

    # X = U S V^T. Rows of V^T are the PC directions and U S are the scores.
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    # Centering makes rank(X) <= n_images - 1; discard the final numerical null PC.
    n_components = min(images.shape[0] - 1, images.shape[1])
    singular_values = singular_values[:n_components]
    components = vt[:n_components]
    scores = u[:, :n_components] * singular_values

    explained_variance = singular_values**2 / (images.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    return mean_image, scores, components, explained_variance_ratio


def save_component_figure(
    mean_image: np.ndarray,
    scores: np.ndarray,
    components: np.ndarray,
    image_shape: tuple[int, int],
    output_path: Path,
) -> None:
    """Show min-score reconstruction, mean, and max-score reconstruction per PC."""
    n_show = min(N_PCS_TO_VISUALISE, components.shape[0])
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 2.75 * n_show), squeeze=False)

    for pc_index in range(n_show):
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

        for column, (axis, image, title) in enumerate(zip(axes[pc_index], images, titles)):
            axis.imshow(np.clip(image.reshape(image_shape), 0, 255), cmap="gray", vmin=0, vmax=255)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    fig.suptitle("Image variation along the first principal components", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_variance_figure(
    explained_ratio: np.ndarray,
    n_selected: int,
    output_path: Path,
) -> None:
    """Plot individual explained variance for every non-null PC."""
    component_numbers = np.arange(1, len(explained_ratio) + 1)
    cumulative = np.cumsum(explained_ratio)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(
        component_numbers,
        explained_ratio * 100,
        width=1.0,
        color="#3973ac",
        label="Variance explained by each PC",
    )
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance (%)", color="#28527a")
    axis.tick_params(axis="y", labelcolor="#28527a")
    axis.set_xlim(0, len(explained_ratio) + 1)

    cumulative_axis = axis.twinx()
    cumulative_axis.plot(
        component_numbers,
        cumulative * 100,
        color="#c43c39",
        linewidth=2,
        label="Cumulative explained variance",
    )
    cumulative_axis.axhline(
        VARIANCE_THRESHOLD * 100,
        color="#c43c39",
        linestyle=":",
        linewidth=1.25,
    )
    cumulative_axis.axvline(
        n_selected,
        color="#333333",
        linestyle="--",
        linewidth=1.25,
        label=f"Selected: first {n_selected} PCs",
    )
    cumulative_axis.set_ylabel("Cumulative explained variance (%)", color="#9c2f2c")
    cumulative_axis.tick_params(axis="y", labelcolor="#9c2f2c")
    cumulative_axis.set_ylim(0, 102)

    handles_1, labels_1 = axis.get_legend_handles_labels()
    handles_2, labels_2 = cumulative_axis.get_legend_handles_labels()
    axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="center right")
    axis.set_title("Explained image variance for all principal components")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_scores(paths: list[Path], scores: np.ndarray, output_path: Path) -> None:
    """Save every image's scores with filenames that can be joined to the ratings."""
    fieldnames = ["filename"] + [f"PC{i}" for i in range(1, scores.shape[1] + 1)]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for path, row in zip(paths, scores):
            writer.writerow([path.name, *[f"{value:.10g}" for value in row]])


def main() -> None:
    paths, images, image_shape = load_images(INPUT_DIR)
    mean_image, scores, components, explained_ratio = compute_pca(images)

    cumulative = np.cumsum(explained_ratio)
    n_selected = int(np.searchsorted(cumulative, VARIANCE_THRESHOLD) + 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_component_figure(
        mean_image,
        scores,
        components,
        image_shape,
        OUTPUT_DIR / "first_five_pcs.png",
    )
    save_variance_figure(
        explained_ratio,
        n_selected,
        OUTPUT_DIR / "explained_variance_all_pcs.png",
    )
    save_scores(paths, scores, OUTPUT_DIR / "pca_scores.csv")

    # Compact machine-readable output needed for reconstruction in later tasks.
    np.savez_compressed(
        OUTPUT_DIR / "pca_model_90pct.npz",
        mean_image=mean_image.astype(np.float32),
        components=components[:n_selected].astype(np.float32),
        explained_variance_ratio=explained_ratio.astype(np.float32),
        image_shape=np.asarray(image_shape),
        n_selected=np.asarray(n_selected),
        variance_threshold=np.asarray(VARIANCE_THRESHOLD),
    )

    print(f"Loaded {len(paths)} images of shape {image_shape} ({images.shape[1]} pixels).")
    print(f"PCA returned {len(explained_ratio)} non-null components.")
    print(
        f"The first {n_selected} PCs explain {cumulative[n_selected - 1] * 100:.2f}% "
        f"of the variance (threshold: {VARIANCE_THRESHOLD * 100:.0f}%)."
    )
    print("First five PCs explain:")
    for index, ratio in enumerate(explained_ratio[:5], start=1):
        print(f"  PC{index}: {ratio * 100:.2f}%")
    print(f"Saved results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
