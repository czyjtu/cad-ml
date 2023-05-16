import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings

warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# torchvision libraries
import torch
import torchvision

[67 / 88]
from torchvision import transforms as torchtrans
from torchvision import transforms as tvt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils

import csv
from pathlib import Path
from typing import Optional, Any
from torchtyping import TensorType


class ObjectDetectionDataset(torch.utils.data.Dataset):
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
            [(int(a['xmin'])/wt)*224, (int(a['ymin'])/ht)*224, (int(a['xmax'])/wt)*224, (int(a['ymax'])/ht)*224]
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


def get_object_detection_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# use our dataset and defined transformations
dataset = ObjectDetectionDataset('external-dataset/train_labels.csv', 'external-dataset/dataset')
dataset_test = ObjectDetectionDataset('external-dataset/test_labels.csv', 'external-dataset/dataset')

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=20, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=20, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# to train on gpu if selected.
device = torch.device('cuda:0')

num_classes = 2

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device).double()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=3.0e-4,
                            momentum=0.9, weight_decay=0.0005)

# training for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

model.save('output/model.pth')