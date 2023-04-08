from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from coronaryx.data import CoronagraphyScan, ROI


def read_dataset(dataset_dir: str):
    dataset_dir = Path(dataset_dir)

    items = []
    for item_dir in dataset_dir.iterdir():
        if not item_dir.is_dir() and not item_dir.name.startswith('seg'):
            continue

        # scan
        with open(item_dir / 'representative_frame.p', 'rb') as f:
            scan = pickle.load(f)

        # centerline
        with open(item_dir / 'centreline.p', 'rb') as f:
            centerline = pickle.load(f)

        # vessel mask
        mask_img = Image.open(item_dir / 'segmentation.png')
        mask = np.array(mask_img)[:, :, 0] == 253

        # roi
        with open(item_dir / 'roi.p', 'rb') as f:
            roi_list = pickle.load(f)

        rois = []
        for d in roi_list:
            roi = ROI(d['start']['x'], d['start']['y'], d['end']['x'], d['end']['y'])
            rois.append(roi)

        # adding
        item = CoronagraphyScan(
            item_dir.name,
            scan,
            mask,
            centerline,
            rois
        )
        items.append(item)
    return items
