
import argparse
import pathlib
import torch
import torch.optim as optim
from torchvision import datasets
from train import train_pix2pix
from net import UnetGenerator, PatchDiscriminator, weights_init
from utils import ABImageDataset, RandHFlipTwoIMG, RandomCropTwoIMG, ResizeTwoIMG, ToTensorTwoIMG, ComposeTwoIMG, Normalize

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
                        type=int, default=1)
    parser.add_argument('-vs', '--val_size', help='validation dataset size. defalut value is 10',
                        type=float, default=0.15)
    parser.add_argument('-m', '--mean', help='mean to use for noarmalization',
                        type=float, default=0.5)
    parser.add_argument('-std', '--std', help='std to use for noarmalization',
                        type=float, default=0.5)
    parser.add_argument('--ngf', help='number of gen filters of first Convolution',
                        type=int, default=64)
    parser.add_argument('--ndf', help='number of dis filters of first Convolution',
                        type=int, default=64)
    parser.add_argument('--norm_type', help='specify normalization type. defalut value is `batch`,'
                        'batch: Batch Normalization, instance: Instance Normalization, none: don\'t apply normalization',
                        choices=['batch', 'instance', 'none'], default='batch')
    parser.add_argument('--lambda_L1', type=float, default=100.0,
                        help='weight for L1 loss. default is 100')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_worker for Dataloader')
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0,'
                        ' -1 is means don\'t use gpu',
                        choices=[-1, 0, 1], type=int, default=0)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)
    # parse arguments
    opt = parser.parse_args()
    out = pathlib.Path(
        "result_{0}/result_{0}_{1}".format(opt.number, opt.seed))
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
    train_data_dir = pathlib.Path('../data/pix2pix/facades/train').resolve()
    val_data_dir = pathlib.Path('../data/pix2pix/facades/val').resolve()
    # transform
    transform = {
        'train': ComposeTwoIMG([
            ResizeTwoIMG(286),
            RandHFlipTwoIMG(p=0.5),
            RandomCropTwoIMG(256),
            ToTensorTwoIMG()
        ]),
        'val': ComposeTwoIMG([
            ToTensorTwoIMG()])
    }
    # load datasets
    mean = [opt.mean, opt.mean, opt.mean]
    std = [opt.std, opt.std, opt.std]
    datasets = {
        'train': ABImageDataset(root=train_data_dir, transform=transform['train'],
                                normalizer=Normalize(mean, std)),
        'val': ABImageDataset(root=val_data_dir, transform=transform['val'],
                              val_size=9, normalizer=Normalize(mean, std))
    }

    # build model gen, dis
    models = {
        'gen': UnetGenerator(ngf=opt.ngf),
        'dis': PatchDiscriminator(ndf=opt.ndf)
    }
    # initialize models parameters
    for model in models.values():
        model.apply(weights_init)

    # define optimizers
    def make_optimizer(model, lr=0.0002, beta1=0.5):
        optimizer = optim.Adam(params=model.parameters(),
                               lr=lr, betas=(beta1, 0.999))
        return optimizer

    optimizers = {
        'gen': make_optimizer(models['gen']),
        'dis': make_optimizer(models['dis'])
    }

    print("train dir: {} | val dir{}".format(train_data_dir, val_data_dir))
    print("train size: {} | val size: {}".format(
        len(datasets['train']), len(datasets['val'])))

    train_pix2pix(models, datasets, optimizers=optimizers, lam=opt.lambda_L1,
                  num_epochs=opt.epoch, batch_size=opt.batch_size, device=device, out=out, num_workers=opt.num_workers)
