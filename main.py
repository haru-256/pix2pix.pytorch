import argparse
import pathlib
import torch
import torch.optim as optim
from torch.backends import cudnn
import random
import numpy as np
from train import train_pix2pix
from net import UnetGenerator, PatchDiscriminator, weights_init
from utils import ABImageDataset, RandHFlipTwoIMG, RandomCropTwoIMG, SplitImage
from utils import ResizeTwoIMG, ToTensorTwoIMG, ComposeTwoIMG, Normalize, plot_loss


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
    parser.add_argument('-vs', '--val_size', help='validation dataset size. defalut value is 16',
                        type=int, default=16)
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
    parser.add_argument('--dataset', default='facade',
                        choices=['facades', 'edges2shoes', 'edges2handbags'], help='what is datasets to use. default is "facades"')
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
        "{0}/result_{1}/result_{1}_{2}".format(opt.dataset, opt.number, opt.seed))

    # set seed
    # cudnn.deterministic = True # don't use cudnn
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # make directory
    cdir = pathlib.Path('.').resolve()
    for i, path in enumerate(out.parts):
        cdir = cdir / path
        if not cdir.exists():
            cdir.mkdir()

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
    train_data_dir = pathlib.Path(
        '../data/pix2pix/{}/train'.format(opt.dataset)).resolve()
    val_data_dir = pathlib.Path(
        '../data/pix2pix/{}/val'.format(opt.dataset)).resolve()
    # transform
    if opt.dataset == "facades":
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
        right_is_A = True
    else:
        transform = {
            'train': ComposeTwoIMG([
                ToTensorTwoIMG()
            ]),
            'val': ComposeTwoIMG([
                ToTensorTwoIMG()])
        }
        right_is_A = False
    # load datasets
    mean = [opt.mean, opt.mean, opt.mean]
    std = [opt.std, opt.std, opt.std]
    datasets = {
        'train': ABImageDataset(root=train_data_dir, transform=transform['train'],
                                normalizer=Normalize(mean, std),
                                spliter=SplitImage(right_is_A=right_is_A)),
        'val': ABImageDataset(root=val_data_dir, transform=transform['val'],
                              val_size=opt.val_size, normalizer=Normalize(
                                  mean, std),
                              spliter=SplitImage(right_is_A=right_is_A))
    }

    # build model gen, dis
    models = {
        'gen': UnetGenerator(ngf=opt.ngf, norm_type=opt.norm_type),
        'dis': PatchDiscriminator(ndf=opt.ndf, norm_type=opt.norm_type)
    }
    # print("U-Net:\n", models['gen'])
    # print("Discriminator:\n", models['dis'])
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

    log = train_pix2pix(models, datasets, optimizers=optimizers, lam=opt.lambda_L1,
                        num_epochs=opt.epoch, batch_size=opt.batch_size, device=device,
                        out=out, num_workers=opt.num_workers, opt=opt)
    plot_loss(log, out / 'loss_{}_{}.png'.format(opt.number, opt.seed))
