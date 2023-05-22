from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import cv2

import coronaryx as cx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_dataset_dir', help='Path to the dataset directory')
    parser.add_argument('val_dataset_dir', help='Path to the dataset directory')
    parser.add_argument('test_dataset_dir', help='Path to the dataset directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    yaml_conf = f"""
    path: {output_dir}
    train: 'images/train'
    val: 'images/val'
    test: 'images/test'
    
    names:
        0: stenosis
    """

    with open('detection.yaml', 'w') as f:
        f.write(yaml_conf)

    for dataset_dir, part in zip([args.train_dataset_dir, args.val_dataset_dir, args.test_dataset_dir],
                                 ['train', 'val', 'test']):

        dataset = cx.read_dataset(dataset_dir)

        for item in dataset:
            # create images directory
            (output_dir / 'images' / part).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / 'images' / part / f'{item.name}.jpg'), item.scan)

            # create labels directory
            (output_dir / 'labels' / part).mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'labels' / part / f'{item.name}.txt', 'w') as f:
                for roi in item.rois:
                    x1, x2 = roi.start_x / item.scan.shape[0], roi.end_x / item.scan.shape[0]
                    y1, y2 = roi.start_y / item.scan.shape[1], roi.end_y / item.scan.shape[1]

                    f.write(f'0 {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}\n')
