from typing import List

from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch as th
from collections import Counter
from coronaryx.data import CoronagraphyScan, ROI


def read_dataset(dataset_dir: str | Path) -> list[CoronagraphyScan]:
    dataset_dir = Path(dataset_dir)

    items = []
    for item_dir in dataset_dir.iterdir():
        if not item_dir.is_dir() and not item_dir.name.startswith("seg"):
            continue

        # scan
        with open(item_dir / "representative_frame.p", "rb") as f:
            scan = pickle.load(f)

        # centerline
        with open(item_dir / "centreline.p", "rb") as f:
            centerline = pickle.load(f)

        # vessel mask
        mask_img = np.array(Image.open(item_dir / "segmentation.png"))
        if mask_img.ndim == 3:
            mask = mask_img[:, :, 0] == 253
        else:
            mask = mask_img.astype(bool)

        # roi
        with open(item_dir / "roi.p", "rb") as f:
            roi_list = pickle.load(f)

        rois = []
        for d in roi_list:
            roi = ROI(d["start"]["x"], d["start"]["y"], d["end"]["x"], d["end"]["y"])
            rois.append(roi)

        # adding
        item = CoronagraphyScan(item_dir.name, scan, mask, centerline, rois)
        items.append(item)
    return items


def save_dataset(dataset: list[CoronagraphyScan], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)

    for item in dataset:
        item_dir = output_dir / item.name
        item_dir.mkdir(parents=True, exist_ok=True)

        # scan
        with open(item_dir / "representative_frame.p", "wb") as f:
            pickle.dump(item.scan, f)

        # centerline
        with open(item_dir / "centreline.p", "wb") as f:
            pickle.dump(item.centerline, f)

        # vessel mask
        Image.fromarray(item.vessel_mask).save(item_dir / "segmentation.png")

        # roi
        roi_list = []
        for roi in item.rois:
            roi_list.append(
                {
                    "start": {"x": roi.start_x, "y": roi.start_y},
                    "end": {"x": roi.end_x, "y": roi.end_y},
                }
            )
        with open(item_dir / "roi.p", "wb") as f:
            pickle.dump(roi_list, f)


def train_dev_test_split(
    dataset: list[CoronagraphyScan], dev_size: float = 0.15, test_size: float = 0.15
) -> tuple[list[CoronagraphyScan], list[CoronagraphyScan], list[CoronagraphyScan]]:
    assert 0.0 <= dev_size < 1.0
    assert 0.0 <= test_size < 1.0
    assert dev_size + test_size < 1.0

    positive_n = sum([len(item.rois) for item in dataset])

    dev_size = int(dev_size * positive_n)
    test_size = int(test_size * positive_n)
    train_size = positive_n - dev_size - test_size

    train_dataset: List[CoronagraphyScan] = []
    dev_dataset: List[CoronagraphyScan] = []
    test_dataset: List[CoronagraphyScan] = []

    for item in dataset:
        if train_size > 0:
            train_dataset.append(item)
            train_size -= len(item.rois)
        elif dev_size > 0:
            dev_dataset.append(item)
            dev_size -= len(item.rois)
        else:
            test_dataset.append(item)
            test_size -= len(item.rois)

    return train_dataset, dev_dataset, test_dataset


def count_votes(
    model: th.nn.Module,
    images: th.Tensor,
    anchors: list[list[tuple[int, int]]],
    batch: int = 32,
) -> Counter[tuple[int, int]]:
    """
    Args:
        model: a trained model
        images: a batch of images
        anchors: list of nachors inside each image
        batch: batch size to evaluate model
    Returns:
        votes: a batch of votes, one for each anchor
    """
    positive_anchors = np.empty((0, 2), dtype=int)
    anchors_array = [np.array(a, dtype=int) for a in anchors]
    for i in range(0, len(images), batch):
        batch_images = images[i : i + batch]
        batch_anchors = anchors_array[i : i + batch]
        batch_logits = model(batch_images)
        batch_classes = th.argmax(batch_logits, dim=1).detach().numpy()
        for i in np.argwhere(batch_classes == 1).flatten():
            positive_anchors = np.concatenate(
                [positive_anchors, batch_anchors[i]], dtype=int
            )
    return Counter(map(tuple, positive_anchors))
