import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import argparse
import pathlib
import numpy as np
from train import train_pix2pix
from net import UnetGenerator, PatchDiscriminator, weights_init
import datetime


if __name__ == '__main__':
    # make parser
    parser = argparse.ArgumentParser(
        prog='pix2pix',
        usage='`python main.py` for training',
        description='train pix2pix with facade datasets',
        epilog='end',
        add_help=True
    )

    # add argument
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=128)
    parser.add_argument('-vs', '--val_size', help='validation dataset size. defalut value is 10',
                        type=float, default=0.15)
    parser.add_argument('--norm_type', help='specify normalization type. defalut value is `batch`,'
                        'batch: Batch Normalization, instance: Instance Normalization, none: don\'t apply normalization',
                        choices=['batch', 'instance', 'none'], default='batch')
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0,'
                        ' -1 is means don\'t use gpu',
                        choices=[-1, 0, 1], type=int, default=0)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)
    # parse arguments
    opt = parser.parse_args()
    out = pathlib.Path("result_{0}/result_{0}_{1}".format(number, seed))
    # make directory
    pre = pathlib.Path(out.parts[0])
    for i, path in enumerate(out.parts):
        path = pathlib.Path(path)
        if i != 0:
            pre /= path
        if not pre.exists():
            pre.mkdir()
        pre = path

    # put arguments into file
    with open(out / "args.txt", "w") as f:
        f.write(str(opt))
    print('arguments:', opt)

    if opt.gpu == 0:
        device = torch.device("cuda:0")
    elif opt.gpu == 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    # path to data directory
    data_dir = pathlib.Path('../../data/pix2pix/facade/train').resolve()
    # transform
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([127.5, 127.5, 127.5],
                                 [127.5, 127.5, 127.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([127.5, 127.5, 127.5],
                                 [127.5, 127.5, 127.5])])
    }
    # load datasets
    image_datasets = {x: datasets.ImageFolder(data_dir / x,
                                              transform=data_transform[x])
                      for x in ['train', 'val', 'test']}
