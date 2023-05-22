from argparse import ArgumentParser

import numpy as np
import cv2
from skimage.filters import unsharp_mask

import coronaryx as cx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='Path to the dataset directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    args = parser.parse_args()

    dataset = cx.read_dataset(args.dataset_dir)

    for item in dataset:
        image = item.scan[:, :, 0]
        sharpened_image = unsharp_mask(image, radius=5, amount=1) * 255

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        item.scan = clahe.apply(sharpened_image.astype('uint8'))

    cx.save_dataset(dataset, args.output_dir)
