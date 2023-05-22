from argparse import ArgumentParser
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

import coronaryx as cx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='Path to the dataset directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    parser.add_argument('--denoiser', help='Path to the denoiser script')
    # TODO can add hyperparameters for the denoising here
    args = parser.parse_args()

    dataset = cx.read_dataset(args.dataset_dir)

    with tempfile.TemporaryDirectory() as dirname:
        dirname = Path(dirname)

        for item in dataset:
            image_path = dirname / f'{item.name}.png'
            processed_image_path = dirname / f'{item.name}-processed.png'
            Image.fromarray(item.scan).save(image_path)

            status = subprocess.run(
                [args.denoiser, image_path, 'rof', '-l', '0.20', '-i', '20', '-o', processed_image_path]
            )
            if status.returncode != 0:
                print(status.stderr)
                raise RuntimeError('denoising failed')

            processed_image = Image.open(processed_image_path).resize(item.scan.shape, Image.BILINEAR)
            item.scan = np.array(processed_image)[:, :, 0].astype('uint8')

        cx.save_dataset(dataset, args.output_dir)
