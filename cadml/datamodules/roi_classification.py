import random
from typing import Optional, Iterable, Any

import numpy as np
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning.utilities import rank_zero_info

import coronaryx as cx
import coronaryx.algorithms as cxa
import coronaryx.functions as cxf


class ROIClassificationDataset(Dataset):
    def __init__(
            self,
            dataset: list[cx.VesselBranch],
            size: int = 32,
            required_positive_overlap: float = 0.5,  # FIXME not used currently! do we need it? is anchor enough?
            upsample_positive_examples: bool = False,
            train_transform: bool = True,
            transform_size_mean: float = 0.0,
            transform_size_std: float = 3.0,
    ):
        """
        PyTorch Dataset based on the angiograms, extracting images along the vessels, labeling stenosis,
        and applying data augmentation.

        :param dataset: list of vessel branches objects
        :param size: size of the output images
        :param required_positive_overlap: minimum overlap with positive ROI required to be considered a positive example
        :param upsample_positive_examples: upsample the number of positive examples to the number of negative ones
        :param train_transform: apply data augmentation techniques
        :param transform_size_mean: mean of generated eps (we cut out picture of size `size + eps` from the original
                                                            and then resize it to `size`)
        :param transform_size_std: std generated eps (we cut out picture of size `size + eps` from the original
                                                            and then resize it to `size`)
        """

        self.original_dataset = dataset
        self.train_transform = train_transform

        self.size = size

        self.transform_size_mean = transform_size_mean
        self.transform_size_std = transform_size_std

        self.dataset = [
            {
                'branch_obj': branch,  # VesselBranch
                'anchor': anchor,  # np.ndarray['2', float]
                'angle': 0.0,  # float
                'size': size,  # int
                'mirrored': False,  # bool
                # TODO another transformations? background tweaking? noise?
            }
            for branch in dataset
            for anchor in cxa.traverse_branch_nodes(branch)
        ]

        self.upsample_positive_examples = upsample_positive_examples
        self.positive_examples = [
            anchor
            for anchor in self.dataset
            if anchor['branch_obj'].scan.in_any_roi(anchor['anchor'])
        ]
        self.negative_examples = [
            anchor
            for anchor in self.dataset
            if not anchor['branch_obj'].scan.in_any_roi(anchor['anchor'])
        ]

    def _transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            'branch_obj': item['branch_obj'],
            'anchor': item['anchor'],
            'angle': np.random.uniform(0, 2 * np.pi),  # TODO what if the angle will always match the angle of vector (vessel will lay horizontally)?
            'size': item['size'] + self.transform_size_mean + self.transform_size_std * np.random.randn(),
            'mirrored': np.random.random() < 0.5,
        }

    def _materialize_item(self, item: dict[str, Any]) -> tuple[TensorType["in_channels", "height", "width"], int]:
        scan: cx.CoronagraphyScan = item['branch_obj'].scan
        image = np.copy(scan.scan) / 255.0  # normalizing

        # creating meshgrid of points (corresponding with each pixel in the output image)
        dx = np.arange(self.size) - self.size / 2 + 0.5
        dy = np.arange(self.size) - self.size / 2 + 0.5

        # scaling image
        dx = dx * (item['size'] / self.size)
        dy = dy * (item['size'] / self.size)

        # mirrored
        if item['mirrored']:
            dy = dy[::-1]

        # applying rotation (derived from the cos(a + b) and sin(a + b) equations)
        XX, YY = np.meshgrid(dx, dy, indexing='ij')

        sin_alpha = np.sin(item['angle'])
        cos_alpha = np.cos(item['angle'])

        XXA = XX * cos_alpha - YY * sin_alpha
        YYA = YY * cos_alpha + XX * sin_alpha

        rotated = cxf.interp_matrix(image, item['anchor'][0] + XXA, item['anchor'][1] + YYA)
        label = int(scan.in_any_roi(item['anchor']))

        # `.unsqueeze(0)` for the `in_channels`
        return torch.tensor(rotated, dtype=torch.float32).unsqueeze(0), label

    def __len__(self) -> int:
        if self.upsample_positive_examples:
            return 2 * len(self.negative_examples)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[TensorType["in_channels", "height", "width"], int]:
        # this ensures that each example will appear at least once in the epoch
        if idx < len(self.dataset):
            item = self.dataset[idx]
        # for all the rest we sample from the positive examples
        else:
            item = random.choice(self.positive_examples)

        if self.train_transform:
            item = self._transform(item)
        return self._materialize_item(item)


class ROIClassificationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            cfg: DictConfig
    ):
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

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == 'train':
            dataset = cx.read_dataset(self.cfg.datamodule.train_dir)

            self.train_dataset = ROIClassificationDataset([
                branch
                for scan in dataset
                for branch in cxa.split_into_branches(scan)
            ], upsample_positive_examples=self.cfg.datamodule.upsample_positive_examples)

        if stage is None or stage == 'val':
            dataset = cx.read_dataset(self.cfg.datamodule.val_dir)

            self.val_dataset = ROIClassificationDataset([
                branch
                for scan in dataset
                for branch in cxa.split_into_branches(scan)
            ], train_transform=False)

        if stage is None or stage == 'test':
            dataset = cx.read_dataset(self.cfg.datamodule.test_dir)

            self.test_dataset = ROIClassificationDataset([
                branch
                for scan in dataset
                for branch in cxa.split_into_branches(scan)
            ], train_transform=False)

        if stage is None or stage == 'predict':
            dataset = cx.read_dataset(self.cfg.datamodule.predict_dir)

            self.predict_dataset = ROIClassificationDataset([
                branch
                for scan in dataset
                for branch in cxa.split_into_branches(scan)
            ], train_transform=False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=6)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=3)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=3)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset,
                          batch_size=self.predict_batch_size,
                          shuffle=False,
                          num_workers=3)

    # TODO we need collator for sure! make tensors of batches + send them to the corresponding device
    # TODO is default collator enough?
