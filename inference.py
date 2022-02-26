import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset, CustomDatasetSplitByProfile


def load_model(saved_model, device):
    model_cls = getattr(import_module("model"), args['model'])
    model = model_cls(**args['model_args'])

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    transform_module = getattr(import_module("dataset"), args['augmentation'])
    aug_args = args['augmentation_args']
    if aug_args.get('mean', None) is None:
        aug_args['mean'] = args['dataset_args']['mean']
        aug_args['std'] = args['dataset_args']['std']
    val_transform = transform_module(is_train=False, **aug_args)

    dataset = TestDataset(img_paths, val_transform, use_PIL=args['dataset_args']['use_PIL'])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args['valid_batch_size'],
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            outs = model(images)

            if args['dataset_args']['output'] == 'all':
                pred = CustomDatasetSplitByProfile.encode_multi_class(outs)
            elif args['dataset_args']['output'] == 'gender':
                pred = (outs >= .5).squeeze().int()
            else:
                pred = torch.argmax(outs, dim=-1)

            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    import yaml

    parser = argparse.ArgumentParser()
    #
    # # Data and model checkpoints directories
    # parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    # parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    # parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    #
    # # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    # parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    #
    parser.add_argument('--model_dir', type=str)
    model_dir = parser.parse_args().model_dir

    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = args['infer_data_dir']
    output_dir = model_dir

    # os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
