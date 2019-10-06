from options.train_options import TrainOptions
import pathlib
from data.abImageDataset import ABImageDataset
from torch.utils.data import DataLoader
from utils.training import Trainer, Updater
import random
import numpy as np
import torch
from models.pix2pix_model import Pix2PixModel

# option
train_opt = TrainOptions()
opt = train_opt.parse()

# seedの固定
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.device != torch.device("cpu"):
    torch.cuda.manual_seed(opt.seed)

# data path
dataroot = pathlib.Path(opt.dataroot)
trainData_path = (dataroot / "train_df.csv").resolve()
valData_path = (dataroot / "val_df.csv").resolve()
print("Train Dataset Path :", trainData_path)
print("Val Dataset Path :", valData_path)

# dataset
train_dataset = ABImageDataset(df_path=trainData_path, opt=opt)
train4vis_dataset = ABImageDataset(
    df_path=trainData_path, opt=opt, phase="val", vis=True
)
val4vis_dataset = ABImageDataset(df_path=valData_path, opt=opt, phase="val", vis=True)
print("Train Dataset Size : {}".format(len(train_dataset)))

# dataloader for leaning
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.nThreads,
)

# dataloader for visuzlizing
train4vis_dataloader = DataLoader(
    dataset=train4vis_dataset,
    batch_size=len(train4vis_dataset),
    shuffle=False,
    num_workers=opt.nThreads,
)
val4vis_dataloader = DataLoader(
    dataset=val4vis_dataset,
    batch_size=len(train4vis_dataset),
    shuffle=False,
    num_workers=opt.nThreads,
)
dataloaders4vis = {"train": train4vis_dataloader, "val": val4vis_dataloader}

# make model
model = Pix2PixModel(opt)

# updater
updater = Updater(train_dataloader=train_dataloader, model=model, opt=opt)
# trainer
trainer = Trainer(updater, opt, dataloaders4vis)
# run
trainer.run()
