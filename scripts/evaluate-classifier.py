from argparse import ArgumentParser
from typing import Callable
import random
import tqdm

import torch
import torchvision
from torch import nn

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import coronaryx as cx
import coronaryx.algorithms as cxa


torch.use_deterministic_algorithms(True)
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

class ResNet(nn.Module):
    def __init__(self, preprocess: Callable, model: nn.Module):
        super().__init__()
        self.preprocess = preprocess
        # print(self.preprocess)
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt: dict):
        hparams = ckpt['hyper_parameters']

        assert hparams['model']['name'] == 'resnet'
        version = str(hparams['model']['version']) if 'version' in hparams['model'] else '18'

        if version == '18':
            preprocess = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
            model = torchvision.models.resnet18()
        elif version == '34':
            preprocess = torchvision.models.ResNet34_Weights.IMAGENET1K_V1.transforms()
            model = torchvision.models.resnet34()
        elif version == '50':
            preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
            model = torchvision.models.resnet50()
        else:
            raise ValueError(f'Invalid ResNet version: {version}')

        model.fc = nn.Linear(model.fc.in_features, 1)
        state_dict = {
            k.removeprefix('model.model.'): v
            for k, v in ckpt['state_dict'].items()
            if k not in ['loss.pos_weight']
        }
        model.load_state_dict(state_dict, strict=True)
        return cls(preprocess, model)

    def forward(self, X):
        # print(X)
        X = X.repeat(1, 3, 1, 1)  # increases the number of in_channels to 3
        X = self.preprocess(X)
        # print(X)
        # print('--------------------------------')
        logits = self.model(X)
        # torch.save(X, f'{logits.item()}.pt')
        # import sys
        # sys.exit(0)
        return logits


# def are_chunks_connected(chunk1_mask: np.ndarray, chunk2_mask: np.ndarray, mask: np.ndarray):
#     _, labeled_mask = cv2.connectedComponents(mask)
#     return np.all(labeled_mask[chunk1_mask] == labeled_mask[chunk2_mask])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('model_path', help='Path to model')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--decision_threshold', default=0.5, type=float, help='Decision threshold')
    parser.add_argument('--votes_threshold', default=3, type=int, help='Votes threshold')
    parser.add_argument('--iou_threshold', default=0.2, type=float, help='IoU threshold')
    parser.add_argument('--show', action='store_true', help='Show images')
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = cx.read_dataset(args.dataset_dir)
    dataset = [item for item in dataset if len(item.rois) > 0]
    print('Dataset size: ', len(dataset))

    ckpt = torch.load(args.model_path, map_location=device)
    model = ResNet.from_pretrained(ckpt).to(device).eval()
    model.model.eval()

    hparams = ckpt['hyper_parameters']
    roi_size = hparams['datamodule']['roi_size'] if 'roi_size' in hparams['datamodule'] else 32

    tps = 0
    fps = 0
    fns = 0

    for item in tqdm.tqdm(dataset):
        voting_map = np.zeros_like(item.scan, dtype=np.int32)
        branches = cxa.split_into_branches(item)
        for branch in branches:
            for anchor in cxa.traverse_branch_nodes(branch):
                image = item.crop_at(anchor, roi_size)
                image = torch.tensor(image, dtype=torch.float32, device=device) / 255.0

                # for _ in range(10):
                with torch.no_grad():
                    logit = model(image.unsqueeze(0))
                    score = torch.sigmoid(logit).item()
                    # print(score)

                # print('anchor', anchor)
                # print('logit', logit.item())
                # print('score', score, score >= args.decision_threshold)

                anchor = np.array(anchor, dtype=np.int32)
                if score >= args.decision_threshold:
                    top_left = np.maximum(anchor - 16, 0)
                    bottom_right = np.minimum(anchor + 16, np.array(item.scan.shape[:2]) - 1)
                    voting_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += 1

        voting_map = voting_map * item.vessel_mask
        voting_map = np.clip(voting_map, 0, args.votes_threshold)

        if args.show:
            figure, ax = plt.subplots(1, 1)
            heatmap = np.zeros((item.scan.shape[0], item.scan.shape[1], 3), dtype=np.uint8)
            heatmap[:, :, 0] = voting_map / args.votes_threshold * 255
            # heatmap[:, :, 1] = item.vessel_mask * 255
            heatmap[:, :, 2] = item.scan

            ax.imshow(heatmap)
            for roi in item.rois:
                rect = patches.Rectangle(
                    (roi.start_x, roi.start_y),
                    roi.end_x - roi.start_x,
                    roi.end_y - roi.start_y,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
            plt.show()

        # calculate chunks
        chunks_count, chunks, stats, _ = cv2.connectedComponentsWithStats(
            (voting_map >= args.votes_threshold).astype(np.uint8),
            connectivity=4
        )

        chunk_ids = [chunk_id for chunk_id in range(1, chunks_count) if stats[chunk_id][cv2.CC_STAT_AREA] >= 100]

        for roi in item.rois:
            roi_start_y = int(roi.start_y)
            roi_end_y = int(roi.end_y)
            roi_start_x = int(roi.start_x)
            roi_end_x = int(roi.end_x)

            roi_mask = np.zeros_like(item.vessel_mask, dtype=np.uint8)
            roi_mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = \
                item.vessel_mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

            best_iou = 0.0
            for chunk_id in chunk_ids:
                chunk_mask = (chunks == chunk_id).astype(np.uint8)

                # calculate intersection over union
                intersection = np.sum(chunk_mask * roi_mask)
                union = np.sum(chunk_mask + roi_mask > 0)
                iou = intersection / union

                best_iou = max(iou, best_iou)

            if best_iou >= args.iou_threshold:
                tps += 1
            else:
                fns += 1

        for chunk_id in chunk_ids:
            chunk_mask = (chunks == chunk_id).astype(np.uint8)

            best_iou = 0.0
            for roi in item.rois:
                roi_start_y = int(roi.start_y)
                roi_end_y = int(roi.end_y)
                roi_start_x = int(roi.start_x)
                roi_end_x = int(roi.end_x)

                roi_mask = np.zeros_like(item.vessel_mask, dtype=np.uint8)
                roi_mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = \
                    item.vessel_mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

                # calculate intersection over union
                intersection = np.sum(chunk_mask * roi_mask)
                union = np.sum(chunk_mask + roi_mask > 0)
                iou = intersection / union

                best_iou = max(iou, best_iou)

            if best_iou < args.iou_threshold:
                fps += 1

    recall = tps / (tps + fns) if tps > 0 else 0.0
    precision = tps / (tps + fps) if tps > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall > 0 or precision > 0 else 0.0

    print(f'{tps=}, {fps=}, {fns=}')
    print(f'precision  {precision:.4f}')
    print(f'recall     {recall:.4f}')
    print(f'f1         {f1:.4f}')
