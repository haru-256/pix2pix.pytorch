import json
import datetime
from collections import OrderedDict
import torch
from tqdm import tqdm

# from tqdm import tqdm_notebook as tqdm # for debug


class Trainer(object):
    def __init__(self, updater, opt, dataloaders4vis):
        """
        学習全体を扱うクラス．

        Args:
            updater (Updater): Updater のinstance
            opt (argparse): option of this program
            dataloaders4vis (dict) : 生成画像を可視化するためのデータローダー．keyは'train', 'val'

        """
        self.opt = opt
        self.epoch = self.opt.epoch
        self.updater = updater
        self.dataloaders4vis = dataloaders4vis

        if self.opt.device != torch.device("cpu"):
            torch.backends.cudnn.benchmark = True

    def run(self):
        """execute training loop.

        Returns:
            log (OrderDict): training log. This contains Generator loss and Discriminator Loss
        """
        since = datetime.datetime.now()
        # initialize log
        if self.opt.resume:
            with open(self.opt.log_dir / "log.json") as f:
                log = json.load(f)
        else:
            log = OrderedDict()

        # training loop
        epochs = tqdm(
            range(self.opt.start_epoch, self.opt.start_epoch + self.opt.epoch),
            desc="Epoch",
            unit="epoch",
        )
        for epoch in epochs:
            # itterater process
            losses, models, optimizers = self.updater.update(
                coeff4dis=self.opt.coeff4dis
            )

            # preserve train log
            log["epoch_{}".format(epoch)] = OrderedDict(
                train_dis_loss=losses["dis"], train_gen_loss=losses["gen"]
            )
            if epoch % self.opt.save_freq == 0:
                # モデルを保存
                if not self.opt.no_ganloss:
                    torch.save(
                        {
                            "dis_model_state_dict": models["dis"].state_dict(),
                            "gen_model_state_dict": models["gen"].state_dict(),
                            "dis_optim_state_dict": optimizers["dis"].state_dict(),
                            "gen_optim_state_dict": optimizers["gen"].state_dict(),
                        },
                        self.opt.model_dir / "pix2pix_{}epoch.tar".format(epoch),
                    )
                else:
                    torch.save(
                        {
                            "gen_model_state_dict": models["gen"].state_dict(),
                            "gen_optim_state_dict": optimizers["gen"].state_dict(),
                        },
                        self.opt.model_dir / "pix2pix_{}epoch.tar".format(epoch),
                    )

                # save generate images
                self.updater.model.save_gen_images(
                    epoch, dataloaders4vis=self.dataloaders4vis
                )
                # save log
                with open(self.opt.log_dir / "log.json", "w") as f:
                    json.dump(log, f, indent=4, separators=(",", ": "))

            # print loss
            text = "Epoch: {:>3} | G_GAN: {:.4f} G_L1: {:.4f} | D_Real: {:.4f} D_Fake: {:.4f}".format(
                epoch,
                losses["gen"]["g_gan"],
                losses["gen"]["g_l1"] * self.opt.lambda_l1,
                losses["dis"]["d_real"] * self.opt.coeff4dis,
                losses["dis"]["d_fake"] * self.opt.coeff4dis,
            )
            tqdm.write(text)
            tqdm.write("=" * 80)

        time_elapsed = datetime.datetime.now() - since
        print("Training complete in {}".format(time_elapsed))
        log["elapsed"] = str(time_elapsed)

        # save log
        with open(self.opt.log_dir / "log.json", "w") as f:
            json.dump(log, f, indent=4, separators=(",", ": "))

        return log


class Updater(object):
    def __init__(self, train_dataloader, model, opt):
        """Updater class.

       1 iterarion のパラメータを更新するクラス．

        Args:
            train_dataloader (torch.utils.data.DataLoader): 学習用のdataloader
            model (model): pix2pix
            opt (Namespace): train.py, resumu.pyの引数

        """
        self.train_dataloader = train_dataloader
        self.model = model
        self.opt = opt

    def update(self, coeff4dis=0.5):
        """
        1 iteration のパラメータ更新を行うクラス

        Parameters:
            coeff (float): discriminator のGANLoss にかけられる係数．公式リポジトリでは0.5をかけられていた．

        Returns:
            losses (dict): dictionary of loss of discriminator and generator.
                each key is dis and gen.
            models (dict): dictionary of model of discriminator, generator and encoder.
                each key is dis, gen and en.
            optimizers (dict): dictionary of optimizer of discriminator and generator.
                each key is dis and gen.
        """
        epoch_loss_D = {"d_real": 0.0, "d_fake": 0.0}
        epoch_loss_G = {"g_gan": 0.0, "g_l1": 0.0}
        iteration = tqdm(self.train_dataloader, desc="Iteration", unit="iter")
        for data_dict in iteration:
            # calc loss of Pix2Pix
            loss_dict = self.model(data_dict)

            # calculate final loss
            loss_D = (loss_dict["d_real"] + loss_dict["d_fake"]) * coeff4dis
            loss_G = loss_dict["g_gan"] + loss_dict["g_l1"] * self.opt.lambda_l1

            # backward Generator
            self.model.optimizer_G.zero_grad()
            loss_G.backward()
            self.model.optimizer_G.step()

            # backward Discrimintor
            if not self.opt.no_ganloss:
                self.model.optimizer_D.zero_grad()
                loss_D.backward()
                self.model.optimizer_D.step()

            # ロスを保存
            for epoch_loss in [epoch_loss_D, epoch_loss_G]:
                for key in epoch_loss.keys():
                    if isinstance(loss_dict[key], torch.Tensor):
                        epoch_loss_G[key] += loss_dict[key].item()

        loss_dict = {
            "dis": self.mean(epoch_loss_D, length=len(self.train_dataloader)),
            "gen": self.mean(epoch_loss_G, length=len(self.train_dataloader)),
        }
        if not self.opt.no_ganloss:
            models = {"dis": self.model.netD, "gen": self.model.netG}
            optimizers = {"dis": self.model.optimizer_D, "gen": self.model.optimizer_G}
        else:
            models = {"gen": self.model.netG}
            optimizers = {"gen": self.model.optimizer_G}

        return loss_dict, models, optimizers

    @staticmethod
    def mean(loss_dict, length):
        for key, value in loss_dict.items():
            if value is None:
                continue
            loss_dict[key] = value / length

        return loss_dict
