from argparse import ArgumentParser
from pathlib import Path
from operator import attrgetter
from typing import Optional, Any
from tqdm import trange

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision
import torchvision.transforms as tvt
from torch.utils.data import Dataset
from torchtyping import TensorType

import coronaryx as cx


class CoronaryXObjectDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        transforms: Optional[Any] = None,
    ):

        self.dataset = sorted([
            item
            for item in cx.read_dataset(dataset_dir)
            if item.rois
        ], key=attrgetter('name'))
        self.transforms = transforms or tvt.ToTensor()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[TensorType["in_channels", "height", "width"], dict[str, Any]]:

        item = self.dataset[idx]
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


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument('checkpoint_path', help='Path to the checkpoint to evaluate')
    parser.add_argument('--device', default='cuda', help='Device to use for evaluation')
    parser.add_argument('--iou_nms_threshold', default=0.3, type=float, help='IoU threshold for non-maximum suppression')
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='IoU threshold for evaluation')
    parser.add_argument('--score_threshold', default=0.5, type=float, help='score threshold to filter out all others')
    parser.add_argument('--show_examples', action='store_true', help='show examples with gold annotations and predictions')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = torch.load(args.checkpoint_path, map_location=device)
    model.eval()

    dataset = CoronaryXObjectDetectionDataset(args.data_dir)

    tps = 0
    fps = 0
    fns = 0

    predicted_boxes = []
    predicted_scores = []

    for idx in trange(len(dataset)):
        image, target = dataset[idx]
        image = image.repeat(3, 1, 1)

        with torch.no_grad():
            output = model([image])[0]

        output = apply_nms(output, iou_thresh=args.iou_threshold)

        scores_mask = output['scores'] >= args.score_threshold
        output_boxes = output['boxes'][scores_mask]

        predicted_boxes.append(output_boxes)
        predicted_scores.append(output['scores'][scores_mask])

        if output_boxes.size(0) == 0:
            fns += target['boxes'].size(0)
            continue

        # (i, j) is IoU between target's i-th and output's j-th
        ious = torchvision.ops.box_iou(target['boxes'], output_boxes)

        # true positives and false negatives
        hits = torch.sum(torch.max(ious, dim=1).values >= args.iou_threshold)
        tps += hits
        fns += ious.size(0) - hits

        # false positives
        fps += torch.sum(torch.max(ious, dim=0).values < args.iou_threshold)

    recall = tps / (tps + fns) if tps > 0 else 0.0
    precision = tps / (tps + fps) if tps > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall > 0 or precision > 0 else 0.0

    print(f'{tps=}, {fps=}, {fns=}')
    print(f'precision  {precision:.4f}')
    print(f'recall     {recall:.4f}')
    print(f'f1         {f1:.4f}')

    if args.show_examples:
        for idx in range(len(dataset)):
            image, target = dataset[idx]

            gold_boxes = target['boxes']
            pred_boxes = predicted_boxes[idx]
            scores = predicted_scores[idx]

            fig, ax = plt.subplots()
            ax.imshow(image[0, ...], cmap='gray')
            for box in gold_boxes:
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='b', facecolor='none'
                )
                ax.add_patch(rect)

            for box, score in zip(pred_boxes, scores):
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='g', facecolor='none'
                )
                ax.add_patch(rect)

                ax.text(box[0], box[1]-2, f'{score:.2f}', color='green', fontsize=10, weight='bold')

            plt.show()
