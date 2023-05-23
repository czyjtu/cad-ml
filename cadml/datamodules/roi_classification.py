import random
from typing import Optional, Iterable, Any

import numpy as np
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
import torch as th
import torchvision.transforms as tvt
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import coronaryx as cx
import coronaryx.algorithms as cxa
import coronaryx.functions as cxf
import matplotlib.pyplot as plt


class ROIClassificationDataset(Dataset):
    def __init__(
        self,
        dataset: list[cx.VesselBranch],
        transforms: Optional[Any] = None,
        size: int = 32,
        upsample_positive_examples: bool = False,
        apply_segmentation_mask: bool = False,
    ):
        """
        PyTorch Dataset based on the angiograms, extracting images along the vessels, labeling stenosis,
        and applying data augmentation.

        :param dataset: list of vessel branches objects
        :param transforms: torchvision transforms to be applied to the images
        :param size: size of the output images
        :param upsample_positive_examples: upsample the number of positive examples to the number of negative ones
        """

        self.original_dataset = dataset
        self.size = size
        self.upsample_positive_examples = upsample_positive_examples

        self.items = []
        images = []
        labels = []
        for branch in dataset:
            for anchor in cxa.traverse_branch_nodes(branch):
                item = {
                    "branch_obj": branch,  # VesselBranch
                    "anchor": anchor,  # np.ndarray['2', float]
                    "angle": 0.0,  # float
                    "size": size,  # int
                    "mirrored": False,  # bool
                    # TODO another transformations? background tweaking? noise?
                }
                images.append(
                    branch.scan.crop_at(
                        anchor, size, apply_segmentation_mask=apply_segmentation_mask
                    ) / 255.0
                )
                labels.append(int(branch.scan.in_any_roi(anchor)))
                self.items.append(item)
        self.images = np.array(images, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.byte)
        self.transforms = transforms or tvt.ToTensor()

    def __len__(self) -> int:
        if self.upsample_positive_examples:
            return 2 * (self.labels == 0).sum()
        return len(self.items)

    def __getitem__(
        self, idx: int
    ) -> tuple[TensorType["in_channels", "height", "width"], int]:
        # this ensures that each example will appear at least once in the epoch
        if idx < len(self.items):
            img = self.images[idx]
            label = self.labels[idx]
        # for all the rest we sample from the positive examples
        else:
            label = 1
            img = random.choice(self.images[self.labels == label])

        return self.transforms(img), label


class ROIClassificationDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        self.train_batch_size: int = cfg.datamodule.train_batch_size
        self.val_batch_size: int = cfg.datamodule.val_batch_size
        self.test_batch_size: int = cfg.datamodule.test_batch_size
        self.predict_batch_size: int = cfg.datamodule.predict_batch_size

        self.train_dataset: Optional[ROIClassificationDataset] = None
        self.val_dataset: Optional[ROIClassificationDataset] = None
        self.test_dataset: Optional[ROIClassificationDataset] = None
        self.predict_dataset: Optional[ROIClassificationDataset] = None

    def prepare_data(self) -> None:
        pass

    def get_transforms(self, allow_padding: bool, roi_size: int) -> Any:
        transforms = [
            tvt.ToTensor(),
            tvt.RandomHorizontalFlip(p=0.5),
            tvt.RandomVerticalFlip(p=0.5),
            tvt.RandomRotation(degrees=180),
        ]
        if not allow_padding:
            transforms.append(tvt.CenterCrop(roi_size))
        return tvt.Compose(transforms)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "train":
            transforms = self.get_transforms(
                allow_padding=self.cfg.datamodule.allow_padding,
                roi_size=self.cfg.datamodule.roi_size,
            )
            raw_roi_size = self.cfg.datamodule.roi_size
            if not self.cfg.datamodule.allow_padding:
                raw_roi_size = np.ceil(raw_roi_size * np.sqrt(2)).astype(int)
            dataset = cx.read_dataset(self.cfg.datamodule.train_dir)

            self.train_dataset = ROIClassificationDataset(
                [
                    branch
                    for scan in dataset
                    for branch in cxa.split_into_branches(scan)
                ],
                upsample_positive_examples=self.cfg.datamodule.upsample_positive_examples,
                transforms=transforms,
                size=raw_roi_size,
                apply_segmentation_mask=self.cfg.datamodule.apply_segmentation_mask,
            )

        if stage is None or stage == "val":
            dataset = cx.read_dataset(self.cfg.datamodule.val_dir)

            self.val_dataset = ROIClassificationDataset(
                [
                    branch
                    for scan in dataset
                    for branch in cxa.split_into_branches(scan)
                ],
            )

        if stage is None or stage == "test":
            dataset = cx.read_dataset(self.cfg.datamodule.test_dir)

            self.test_dataset = ROIClassificationDataset(
                [
                    branch
                    for scan in dataset
                    for branch in cxa.split_into_branches(scan)
                ],
            )

        if stage is None or stage == "predict":
            dataset = cx.read_dataset(self.cfg.datamodule.predict_dir)

            self.predict_dataset = ROIClassificationDataset(
                [
                    branch
                    for scan in dataset
                    for branch in cxa.split_into_branches(scan)
                ],
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=2,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=2,
        )
