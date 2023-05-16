import csv
from pathlib import Path
from typing import Optional, Iterable, Any

import cv2
import numpy as np
import torchvision.io
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
import torchvision.transforms as tvt
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import coronaryx as cx


class PascalVOCObjectDetectionDataset(Dataset):
    def __init__(
            self,
            csv_file: str | Path,
            images_folder: str | Path,
            transforms: Optional[Any] = None,
    ):
        """
        PyTorch Dataset based on the angiograms, which are stored in a folder and described in a CSV file.
        The CSV file must contain the following columns: `filename`, `xmin`, `ymin`, `xmax`, `ymax`, `class`.
        The `class` column must contain the class name as a string.
        The `xmin`, `ymin`, `xmax`, `ymax` columns must contain the coordinates of the bounding box of the object in
        the image.
        The coordinates must be integers. The origin is in the top left corner of the image.

        :param csv_file: path to the CSV file with the annotations
        :param images_folder: path to the folder with the images
        :param transforms: torchvision transforms to be applied to the images
        """

        self.images_folder = Path(images_folder)

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            stenosis_annotations = list(reader)

        # annotations per image
        self.annotations = dict()
        for annotation in stenosis_annotations:
            image_name = annotation['filename']
            if image_name not in self.annotations:
                self.annotations[image_name] = []
            self.annotations[image_name].append(annotation)

        self.image_names = sorted(self.annotations.keys())
        self.transforms = transforms or tvt.ToTensor()

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(
            self, idx: int
    ) -> tuple[TensorType["in_channels", "height", "width"], dict[str, Any]]:
        # TODO can be rewritten using XML files https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch

        image_name = self.image_names[idx]
        image_path = self.images_folder / image_name
        annotations = self.annotations[image_name]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wt, ht = image.shape[1], image.shape[0]
        image = cv2.resize(image, (224, 224), cv2.INTER_AREA) / 255.0

        boxes = torch.tensor([
            [(int(a['xmin']) / wt) * 224, (int(a['ymin']) / ht) * 224, (int(a['xmax']) / wt) * 224,
             (int(a['ymax']) / ht) * 224]
            for a in annotations
        ])

        target = {
            'boxes': boxes,
            'labels': torch.ones((len(annotations),), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(annotations),), dtype=torch.int64)
        }

        if self.transforms:
            # TODO boxes are not transformed
            image = self.transforms(image)

        return image.to(torch.float64), target


class CoronaryXObjectDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        transforms: Optional[Any] = None,
    ):

        self.original_dataset = cx.read_dataset(dataset_dir)
        self.transforms = transforms or tvt.ToTensor()

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[TensorType["in_channels", "height", "width"], dict[str, Any]]:

        item = self.original_dataset[idx]
        image = item.scan  # FIXME in_channels?
        wt, ht = image.shape[1], image.shape[0]
        image = cv2.resize(image, (224, 224), cv2.INTER_AREA) / 255.0

        boxes = torch.tensor([
            [roi.start_x / wt * 224, roi.start_y / ht * 224, roi.end_x / wt * 224, roi.end_y / ht * 224]
            for roi in item.rois
        ])

        target = {
            'boxes': boxes,
            'labels': torch.ones((boxes.size(0),), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((boxes.size(0),), dtype=torch.int64)
        }

        if self.transforms:
            # TODO boxes are not transformed
            image = self.transforms(image)

        return image, target


class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_batch_size: int = cfg.datamodule.train_batch_size
        self.val_batch_size: int = cfg.datamodule.val_batch_size
        self.test_batch_size: int = cfg.datamodule.test_batch_size
        self.predict_batch_size: int = cfg.datamodule.predict_batch_size

        self.train_dataset: Optional[ObjectDetectionDataset] = None
        self.val_dataset: Optional[ObjectDetectionDataset] = None
        self.test_dataset: Optional[ObjectDetectionDataset] = None
        self.predict_dataset: Optional[ObjectDetectionDataset] = None

    def prepare_data(self) -> None:
        pass

    def get_transforms(self) -> Any:
        transforms = [
            tvt.ToTensor(),
            tvt.RandomHorizontalFlip(p=0.5),
        ]
        return tvt.Compose(transforms)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "train":
            transforms = self.get_transforms()

            self.train_dataset = ObjectDetectionDataset(
                self.cfg.datamodule.train_csv,
                self.cfg.datamodule.train_dir,
                transforms=transforms,
            )

        if stage is None or stage == "val":
            self.val_dataset = ObjectDetectionDataset(
                self.cfg.datamodule.val_csv,
                self.cfg.datamodule.val_dir,
            )

        if stage is None or stage == "test":
            self.test_dataset = ObjectDetectionDataset(
                self.cfg.datamodule.test_csv,
                self.cfg.datamodule.test_dir,
            )

        if stage is None or stage == "predict":
            self.predict_dataset = ObjectDetectionDataset(
                self.cfg.datamodule.predict_csv,
                self.cfg.datamodule.predict_dir,
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
