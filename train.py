import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import colors
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import f1_score, confusion_matrix
from tqdm import tqdm

from dataset import MaskBaseDataset
from loss import create_criterion
from optim import get_opt_module
from scheduler import get_scheduler_module


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# def grid_image(np_images, gts, preds, n=16, shuffle=False):
#     batch_size = np_images.shape[0]
#     assert n <= batch_size
#
#     choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
#     figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
#     plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
#     n_grid = np.ceil(n ** 0.5)
#     tasks = ["mask", "gender", "age"]
#     for idx, choice in enumerate(choices):
#         gt = gts[choice].item()
#         pred = preds[choice].item()
#         image = np_images[choice]
#         # title = f"gt: {gt}, pred: {pred}"
#         gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
#         pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
#         title = "\n".join([
#             f"{task} - gt: {gt_label}, pred: {pred_label}"
#             for gt_label, pred_label, task
#             in zip(gt_decoded_labels, pred_decoded_labels, tasks)
#         ])
#
#         plt.subplot(n_grid, n_grid, idx + 1, title=title)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(image, cmap=plt.cm.binary)
#
#     return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m is not None]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args['seed'])

    save_dir = increment_path(os.path.join(model_dir, args['name']))
    print(colors.red(f'save dir: {save_dir}'))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args['dataset'])
    train_dataset = dataset_module(
        data_dir=data_dir,
        profiles_file=args['train_profile'],
        **args['dataset_args'],
    )
    val_dataset = dataset_module(
        data_dir=data_dir,
        profiles_file=args['valid_profile'],
        pass_calc_statistics=True,
        **args['dataset_args'],
    )
    num_classes = args['num_classes']  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args['augmentation'])

    aug_args = args['augmentation_args']
    if aug_args.get('mean', None) is None:
        mean = train_dataset.mean
        std = train_dataset.std
        aug_args['mean'] = mean if isinstance(mean, list) else mean.tolist()
        aug_args['std'] = std if isinstance(std, list) else std.tolist()

    train_transform = transform_module(**aug_args)
    val_transform = transform_module(is_train=False, **aug_args)
    train_dataset.set_transform(train_transform)
    val_dataset.set_transform(val_transform)

    # -- data_loader
    # train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args['train_batch_size'],
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args['valid_batch_size'],
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model_module = getattr(import_module("model"), args['model'])  # default: BaseModel
    model = model_module(**args['model_args']).to(device)
    # model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args['criterion'], **args['criterion_args'])  # default: cross_entropy
    # TODO: 멘토님 optimizer 추가
    # opt_module = getattr(import_module("torch.optim"), args['optimizer'])
    opt_module = get_opt_module(args['optimizer'])
    # TODO: 모델 파트별로 optimizer 설정
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )
    # -- lr scheduler
    # TODO: scheduler 파일 분리, cos 구현
    scheduler_module = get_scheduler_module(args['scheduler'])
    scheduler = scheduler_module(optimizer, **args['scheduler_args'])


    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(args, f, allow_unicode=True, indent=2)

    train_report_format = '{:^ 10.4f}{:^ 10.2%}{:^ 10.6f}'

    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf
    save_val_acc = 0
    save_val_f1 = 0
    save_val_loss = 0
    for epoch in range(args['epochs']):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        print(colors.blue(f'Epoch [ {epoch + 1} / {args["epochs"]} ]'))
        print(colors.green(('{:^10}'*3).format('loss', 'acc', 'lr')))
        loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, train_batch in loader_tqdm:
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)

            if args['dataset_args']['output'] == 'all':
                preds = train_dataset.encode_multi_class(outs)
                targets = train_dataset.encode_multi_class(labels)
            elif args['dataset_args']['output'] == 'gender':
                preds = (outs >= .5).squeeze().int()
                targets = labels
                outs = outs.squeeze()
                labels = labels.float()
            else:
                preds = torch.argmax(outs, dim=-1)
                targets = labels

            optimizer.zero_grad()
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == targets).sum().item()
            if (idx + 1) % args['log_interval'] == 0:
                train_loss = loss_value / args['log_interval']
                train_acc = matches / args['train_batch_size'] / args['log_interval']
                current_lr = get_lr(optimizer)
                # print(
                #     f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                #     f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                # )
                report = train_report_format.format(train_loss, train_acc, current_lr)
                loader_tqdm.set_description(report)
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/learning rate", current_lr, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_preds_list = []
            val_targets_list = []
            draw_cm = False
            # figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)

                if args['dataset_args']['output'] == 'all':
                    preds = train_dataset.encode_multi_class(outs)
                    targets = train_dataset.encode_multi_class(labels)
                elif args['dataset_args']['output'] == 'gender':
                    preds = (outs >= .5).squeeze().int()
                    targets = labels
                    outs = outs.squeeze()
                    labels = labels.float()
                else:
                    preds = torch.argmax(outs, dim=-1)
                    targets = labels

                loss_item = criterion(outs, labels).item()
                acc_item = (preds == targets).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_preds_list.append(preds.detach().clone().cpu())
                val_targets_list.append(targets.detach().clone().cpu())

                # if figure is None:
                #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                #     inputs_np = dataset_module.denormalize_image(inputs_np, train_dataset.mean, train_dataset.std)
                #     figure = grid_image(
                #         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                #     )

            val_loss = np.sum(val_loss_items) / len(val_loader)

            val_preds_cat = torch.cat(val_preds_list)
            val_targets_cat = torch.cat(val_targets_list)
            val_f1 = f1_score(val_preds_cat, val_targets_cat, average='macro', num_classes=num_classes)

            val_acc = np.sum(val_acc_items) / len(val_dataset)

            if val_acc > best_val_acc:
                if args['best_criterion'] == 'acc':
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.state_dict(), f"{save_dir}/best.pth")
                    save_val_acc = val_acc
                    save_val_f1 = val_f1
                    save_val_loss = val_loss
                best_val_acc = val_acc
                draw_cm = True
            if val_f1 > best_val_f1:
                if args['best_criterion'] == 'f1':
                    print(f"New best model for val f1 : {val_f1:.4}! saving the best model..")
                    torch.save(model.state_dict(), f"{save_dir}/best.pth")
                    save_val_acc = val_acc
                    save_val_f1 = val_f1
                    save_val_loss = val_loss
                best_val_f1 = val_f1
                draw_cm = True
            if val_loss < best_val_loss:
                if args['best_criterion'] == 'loss':
                    print(f"New best model for val loss : {val_loss:.4}! saving the best model..")
                    torch.save(model.state_dict(), f"{save_dir}/best.pth")
                    save_val_acc = val_acc
                    save_val_f1 = val_f1
                    save_val_loss = val_loss
                best_val_loss = val_loss
                draw_cm = True

            if draw_cm:
                figure = plt.figure(figsize=(10, 10), dpi=100)
                cm = confusion_matrix(val_preds_cat, val_targets_cat, num_classes=num_classes)
                sn.heatmap(cm, annot=True, fmt='d')
                plt.ylabel('target')
                plt.xlabel('pred')
                plt.savefig(f'{save_dir}/cm.png')
                logger.add_figure('Val/confusion_matrix', figure, epoch)
                plt.close()

            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            print(
                f"      {'acc':^9}{'f1':^9}{'loss':^9}\n"
                f"curr |{val_acc:^ 8.2%}|{val_f1:^ 8.4f}|{val_loss:^ 8.4f}|\n"
                f"best |{best_val_acc:^ 8.2%}|{best_val_f1:^ 8.4f}|{best_val_loss:^ 8.4f}|\n"
                f"save |{save_val_acc:^ 8.2%}|{save_val_f1:^ 8.4f}|{save_val_loss:^ 8.4f}|"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            # logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    import yaml
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    # parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    # parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    # parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    # parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    # parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    # parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    #
    # # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # args = parser.parse_args()
    with open('config.yaml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    print(args)

    data_dir = args['data_dir']
    model_dir = args['model_dir']

    train(data_dir, model_dir, args)
