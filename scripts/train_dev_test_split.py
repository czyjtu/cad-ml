from argparse import ArgumentParser
from pathlib import Path

import coronaryx as cx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input', help='path to the input dataset folder')
    parser.add_argument('output', help='path for the output folder')
    parser.add_argument('--dev_size', type=float, default=0.15, help='dev size')
    parser.add_argument('--test_size', type=float, default=0.15, help='test size')
    args = parser.parse_args()

    dataset = cx.read_dataset(args.input)
    train, dev, test = cx.train_dev_test_split(dataset, args.dev_size, args.test_size)

    output_folder = Path(args.output).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    cx.save_dataset(train, output_folder / 'train')
    cx.save_dataset(dev, output_folder / 'dev')
    cx.save_dataset(test, output_folder / 'test')

    print(f'Train size: {len(train)}, {len(train) / len(dataset):.2%}')
    print(f'    positives: {sum([len(item.rois) for item in train])}')

    print(f'Dev size: {len(dev)}, {len(dev) / len(dataset):.2%}')
    print(f'    positives: {sum([len(item.rois) for item in dev])}')

    print(f'Test size: {len(test)}, {len(test) / len(dataset):.2%}')
    print(f'    positives: {sum([len(item.rois) for item in test])}')
