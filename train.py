import pathlib
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


def train_pix2pix(models, datasets, optimizers,
                  num_epochs, batch_size, device, out):
    """
    the training function for pix2pix.

    Parameters
    -----------------
    models: dict
        dictionary that contains generator, discriminator

    datasets: torch.utils.data.Dataset
        dataset of image

    optimizer: dict
        dictionary that contains torch.optim.Optimizer for generator or discriminator

    num_epochs: int
        number of epochs

    batch_size: int
        number of batch size

    device: torch.device

    out: pathlib.Path
        represent output directory
    """
    since = datetime.datetime.now()
    epochs = tqdm(range(num_epochs), desc="Epoch", unit='epoch')
    # construct dataloader
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    dataset_sizes = len(datasets)
    criterion = nn.BCELoss()
    # train loop
    for epoch in epochs:
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
            real_labels = torch.ones(
                inputs.size()[0], device=device)
            fake_labels = torch.zeros(
                inputs.size()[0], device=device)
            labels = {'real': real_labels, 'fake': fake_labels}
            ########################################################
            # (1) Update D network: minimize - 1/ 2 {1/N * log(D(x, y)) + 1/N * log(1 - D(x, G(x, z)))}
            ######################################################
            dis_loss = train_dis(models, optimizers['discriminator'],
                                 inputs, outputs, labels, criterion)
            #######################################################
            # (2) Update G netwrk: minimize - 1/N * log(D(x, G(x, z))) + 1/N * |y - G(x, z)|
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


def train_dis(models, dis_optim, inputs, outputs, labels, criterion):
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

    labels: dict
        dictionary that contains label data coresponding to real or fake

    criterion: torch.BCELoss()

    Return
    --------------
    dis_loss: torch.Tensor
        discriminator loss
    """
    gen, dis = models['generator'], models['discriminator']

    # train disctiminator
    dis.train()
    # zero the parameter gradientsreal
    dis_optim.zero_grad()

    # make noise & forward
    z = gen.make_hidden(data.size()[0]).to(data.device)
    x_fake = gen(z)
    y_fake = dis(x_fake)
    y_real = dis()

    # calc loss for discriminator
    dis_loss = criterion(
        y_real, labels['real']) + criterion(y_fake, labels['fake'])

    # update parameters of discrimminator
    dis_loss.backward()
    dis_optim.step()

    return dis_loss


def train_gen(models, gen_optim, fake_labels, criterion):
    """
    train generator by a iteration
    Parameters
    ---------------
    models: dict
        dictionary that contains generator, discriminator
    gen_optim: torch.optim.Optimizer
        optimizer for generator
    fake_labels: torch.Tensor
        label coresponding to fake image
    criterion: torch.BCELoss()
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
    # make noise & forward
    z = gen.make_hidden(fake_labels.size()[0]).to(fake_labels.device)
    x_fake = gen(z)
    y_fake = dis(x_fake)

    # calc loss for generator
    gen_loss = criterion(y_fake, fake_labels)

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
