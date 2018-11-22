import pathlib
import datetime
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
import json
from utils import visualize


def train_pix2pix(models, datasets, optimizers, lam,
                  num_epochs, batch_size, device, out, num_workers=2, opt=None):
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

    lam: float
        a cofficient for l1 loss.

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

    opt: NameSpace
        arguments
    """
    epochs = tqdm(range(num_epochs), desc="Epoch", unit='epoch')
    phases = ['train', 'val']
    # initialize log
    log = OrderedDict()
    # construct dataloader
    train_dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(datasets['val'], batch_size=len(datasets['val']),
                                                 shuffle=False, num_workers=num_workers)
    dataset_sizes = {phase: len(datasets[phase]) for phase in phases}
    # train loop
    since = datetime.datetime.now()

    # make dir to save model & image
    model_dir = out / "model"
    if not model_dir.exists():
        model_dir.mkdir()
    image_dir = out / "gen_image"
    if not image_dir.exists():
        image_dir.mkdir()

    for model in models.values():
        model.to(device)
        model.train()  # apply Dropout and BatchNorm during both training and inference

    for epoch in epochs:
        iteration = tqdm(train_dataloader,
                         desc="Iteration",
                         unit='iter')
        epoch_dis_loss = 0.0
        epoch_gen_loss = 0.0
        for inputs, outputs in iteration:
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            assert ((-1 <= outputs) * (outputs <= 1)
                    ).all(), "input data to discriminator range is not from -1 to 1. Got: {}".format((outputs.min(), outputs.max()))
            ########################################################
            # (1) Update D network: minimize - 1/ 2 {1/N * log(D(x, y)) + 1/N * log(1 - D(x, G(x, z)))}
            # minimize - 1/2N * {softplus(-D(x, y)) + softplus(D(x, y))}
            ######################################################
            dis_loss = train_dis(models, optimizers['dis'],
                                 inputs, outputs)
            #######################################################
            # (2) Update G netwrk: minimize - 1/N * log(D(x, G(x, z))) + 1/N * lambda * |y - G(x, z)|
            # minimize - 1/N { softplus(-D(x, G(x, z))) + lambda * |y - G(x, z)|}
            ######################################################
            gen_loss = train_gen(
                models, optimizers['gen'],
                inputs, outputs, lam)
            # statistics
            epoch_dis_loss += dis_loss.item() * inputs.size()[0]
            epoch_gen_loss += gen_loss.item() * inputs.size()[0]

        # print loss
        epoch_dis_loss /= dataset_sizes['train']
        epoch_gen_loss /= dataset_sizes['train']

        # preserve train log & print train loss
        log["epoch_{}".format(epoch+1)] = OrderedDict(train_dis_loss=epoch_dis_loss,
                                                      train_gen_loss=epoch_gen_loss)
        tqdm.write('Epoch: {} GenLoss: {:.4f} DisLoss: {:.4f}'.format(
            epoch+1, epoch_gen_loss, epoch_dis_loss))
        tqdm.write("-"*60)
        # save model &  by epoch
        torch.save({
            'epoch': epoch,
            'dis_model_state_dict': models['dis'].state_dict(),
            'gen_model_state_dict': models['gen'].state_dict(),
            'dis_optim_state_dict': optimizers['dis'].state_dict(),
            'gen_optim_state_dict': optimizers['gen'].state_dict(),
            'dis_loss': epoch_dis_loss,
            'gen_loss': epoch_gen_loss,
        }, model_dir / 'pix2pix_{}epoch.tar'.format(epoch+1))

        # generate fake image
        visualize(epoch+1, models['gen'], val_dataloader=val_dataloader,
                  log_dir=image_dir, device=device, mean=opt.mean, std=opt.std)

    time_elapsed = datetime.datetime.now() - since
    tqdm.write('Training complete in {}'.format(time_elapsed))

    # save log
    with open(out / "log.json", "w") as f:
        json.dump(log, f, indent=4, separators=(',', ': '))

    return log


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
    gen, dis = models['gen'], models['dis']

    # zero the parameters gradientsreal
    dis_optim.zero_grad()

    # forward
    outputs_fake = gen(inputs)
    assert ((-1 <= outputs_fake) * (outputs_fake <= 1)
            ).all(), "input data to discriminator range"
    " is not from -1 to 1. Got: {}".format(
        (outputs_fake.min(), outputs_fake.max()))
    # stop backprop to the generator by detaching outputs_fake
    fake = dis(outputs_fake.detach(), inputs)
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
    gen, dis = models['gen'], models['dis']
    # zero the parameter gradientsreal
    gen_optim.zero_grad()
    outputs_fake = gen(inputs)
    assert ((-1 <= outputs_fake) * (outputs_fake <= 1)
            ).all(), "input data to discriminator range"
    " is not from -1 to 1. Got: {}".format(
        (outputs_fake.min(), outputs_fake.max()))
    fake = dis(outputs_fake, inputs)

    # calc loss for generator
    gen_loss = torch.mean(F.softplus(-fake)) + lam * \
        F.l1_loss(outputs_fake, outputs)

    # update parameters of discrimminator
    gen_loss.backward()
    gen_optim.step()

    return gen_loss
