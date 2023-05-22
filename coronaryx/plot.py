from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from coronaryx.data import CoronagraphyScan, VesselBranch


def plot_dataset(
    dataset: list[CoronagraphyScan], figsize: tuple[int, int] = (8, 8)
) -> None:
    n_items = len(dataset)
    n_rows = 10
    n_cols = n_items // 10 + 1
    fig, axes = plt.subplots(n_rows, n_cols)

    i = 0
    for axes_row in axes:
        for ax in axes_row:
            if i < n_items:
                ax.imshow(dataset[i].scan, cmap="gray", extent=None)
            else:
                ax.imshow(np.zeros_like(dataset[0].scan), cmap="gray", extent=None)
            ax.axis("off")
            i += 1

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05
    )
    fig.set_size_inches(*figsize)
    plt.show()


def plot_scan(scan: CoronagraphyScan):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    ax1.imshow(scan.scan, cmap="gray")
    ax2.imshow(scan.vessel_mask, cmap="gray")

    nodes = np.array(scan.centerline.nodes)
    ax2.scatter(nodes[:, 1], nodes[:, 0], s=10, c="blue")

    for (x1, y1), (x2, y2) in scan.centerline.edges:
        ax2.plot([y1, y2], [x1, x2], c="pink")

    for roi in scan.rois:
        rect = patches.Rectangle(
            (roi.start_x, roi.start_y),
            roi.end_x - roi.start_x,
            roi.end_y - roi.start_y,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax2.add_patch(rect)

    plt.show()


def plot_branch(branch: VesselBranch):
    scan_copy = deepcopy(branch.scan)
    scan_copy.centerline = branch.branch
    plot_scan(scan_copy)


def plot_votes(votes: Counter[tuple[int, int]], image: np.ndarray, size: int = 32):
    """
    Args:
        votes: Counter of votes for each anchor
        image: the image to display anchors on top of
        size: size of positive anchor annotation
    """
    heatmap = np.zeros_like(image)
    for (row, col), count in votes.items():
        heatmap[
            row - size // 2 : row + size // 2, col - size // 2 : col + size // 2
        ] += count
    # assert np.any(heatmap > 0)

    plt.imshow(image, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.show()
