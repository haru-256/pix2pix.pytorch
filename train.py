import pathlib
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import json


def train_pix2pix(models, datasets, optimizers,
                  num_epochs, batch_size, device, out, num_workers=2):
    """
    the training function for pix2pix.

    Parameters
    -----------------
    models: dict
        dictionary that contains generator, discriminator

    datasets: dict of torch.utils.data.Dataset
        dataset of train image, test image

    optimizer: dict
        dictionary that contains torch.optim.Optimizer for generator or discriminator

    num_epochs: int
        number of epochs

    batch_size: int
        number of batch size

    device: torch.device

    out: pathlib.Path
        represent output directory

    num_workers: int
        how many subprocesses to use for data loading.
         0 means that the data will be loaded in the main process. (default: 2)
    """
    epochs = tqdm(range(num_epochs), desc="Epoch", unit='epoch')
    phases = ['train', 'val']
    # construct dataloader
    dataloader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size,
                                                     shuffle=(phase == 'train'), num_workers=num_workers)
                  for phase in ['train', 'val']}
    dataset_sizes = {phase: len(datasets[phase]) for phase in phases}
    # train loop
    since = datetime.datetime.now()
    [for model in models.item]
    for epoch in epochs:
        for phase in phases:
            iteration = tqdm(dataloader,
                             desc="Iteration",
                             unit='iter')
            epoch_dis_loss = 0.0
            epoch_gen_loss = 0.0
            for inputs, outputs in iteration:
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                assert ((inputs > -1) * (inputs < 1)
                        ).all(), "input data to discriminator range is not from -1 to 1"
                ########################################################
                # (1) Update D network: minimize - 1/ 2 {1/N * log(D(x, y)) + 1/N * log(1 - D(x, G(x, z)))}
                # minimize - 1/2N * {softplus(-D(x, y)) + softplus(D(x, y))}
                ######################################################
                dis_loss = train_dis(models, optimizers['discriminator'],
                                     inputs, outputs, labels, criterion)
                #######################################################
                # (2) Update G netwrk: minimize - 1/N * log(D(x, G(x, z))) + 1/N * lambda * |y - G(x, z)|
                # minimize - 1/N { softplus(-D(x, G(x, z))) + lambda * |y - G(x, z)|}
                ######################################################
                fake_labels = torch.ones_like(
                    fake_labels, device=device)
                gen_loss = train_gen(
                    models, optimizers['generator'],
                    inputs, fake_labels, criterion)
                # statistics
                epoch_dis_loss += dis_loss.item() * inputs.size()[0]
                epoch_gen_loss += gen_loss.item() * inputs.size()[0]

        # print loss
        epoch_dis_loss /= dataset_sizes
        epoch_gen_loss /= dataset_sizes
        tqdm.write('Epoch: {} GenLoss: {:.4f} DisLoss: {:.4f}'.format(
            epoch, epoch_gen_loss, epoch_dis_loss))

        # generate fake image
        visualize(epoch, models['generator'], log_dir=out, device=device)

    time_elapsed = datetime.datetime.now() - since
    tqdm.write('Training complete in {}'.format(time_elapsed))


def train_dis(models, dis_optim, inputs, outputs):
    """
    train discriminator by a iteration
    Parameters
    ---------------
    models: dict
        dictionary that contains generator, discriminator

    dis_optim: torch.optim.Optimizer
        optimizer for discriminator

    inputs: torch.Tensor
        batch data of input image

    outputs: torch.Tensor
        batch data of output image

    Return
    --------------
    dis_loss: torch.Tensor
        discriminator loss
    """
    gen, dis = models['generator'], models['discriminator']

    # train disctiminator
    dis.train()
    # zero the parameters gradientsreal
    dis_optim.zero_grad()

    # forward
    outputs_fake = gen(inputs)
    fake = dis(outputs_fake, inputs)
    real = dis(outputs, inputs)

    # calc loss for discriminator
    dis_loss = torch.mean(F.softplus(-real) + F.softplus(fake)) * 0.5

    # update parameters of discrimminator
    dis_loss.backward()
    dis_optim.step()

    return dis_loss


def train_gen(models, gen_optim, inputs, outputs, lam):
    """
    train generator by a iteration

    Parameters
    ---------------
    models: dict
        dictionary that contains generator, discriminator

    gen_optim: torch.optim.Optimizer
        optimizer for generator

    inputs: torch.Tensor
        batch data of input image

    outputs: torch.Tensor
        batch data of output imag

    lam: float
        cofficient for l1 loss

    Return
    --------------
    gen_loss: torch.Tensor
        generator loss
    """
    gen, dis = models['generator'], models['discriminator']
    # train generator
    gen.train()
    # zero the parameter gradientsreal
    gen_optim.zero_grad()
    outputs_fake = gen(inputs)
    fake = dis(outputs_fake, inputs)

    # calc loss for generator
    gen_loss = torch.mean(F.softplus(-fake)) + lam * \
        F.l1_loss(outputs_fake, outputs)

    # update parameters of discrimminator
    gen_loss.backward()
    gen_optim.step()

    return gen_loss


def visualize(epoch, gen, nrow=7, ncol=7, log_dir=None, device=None):
    """
    visualize generator images
    Parmameters
    -------------------
    epoch: int
        number of epochs
    gen: torch.nn.Module
        generator model
    nrow: int
    ncol: int
    log_dir: pathlib.Path
        path to output directory
    device: torch.device
    """
    gen.eval()
    pre = pathlib.Path(log_dir.parts[0])
    for i, path in enumerate(log_dir.parts):
        path = pathlib.Path(path)
        if i != 0:
            pre /= path
        if not pre.exists():
            pre.mkdir()
        pre = path

    # 生成のもとになる乱数を生成
    np.random.seed(seed=0)
    with torch.no_grad():
        z = gen.make_hidden(nrow*ncol).to(device)
        # Generatorでサンプル生成
        samples = gen(z).cpu()
    np.random.seed()

    images = torchvision.utils.make_grid(samples, normalize=True, nrow=nrow)
    plt.imshow(images.numpy().transpose(1, 2, 0), cmap=plt.cm.gray)
    plt.axis("off")
    plt.title("Epoch: {}".format(epoch))
    plt.savefig(log_dir / "epoch{}.png".format(epoch))
