from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm, trange

from .dataloaders import Datamodule


class Trainer:
    def __init__(self, model, datamodule: Datamodule, lr: float, results_dir: Path, device: torch.device, tqdm_pos=0):
        self.device = device
        self.tqdm_pos = tqdm_pos

        # data
        self.n_classes = datamodule.n_classes
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = None if datamodule.val_set is None else datamodule.val_dataloader()

        # model
        self.model = model.to(device)
        self.optim = AdamW(self.model.parameters(), lr=lr)
        self.sched = ExponentialLR(self.optim, gamma=0.97)

        # loss functions
        class_weights = datamodule.class_weights.to(self.device)
        self.criterion = lambda log_probs, labels: F.nll_loss(log_probs, labels, weight=class_weights)

        # bookkeeping
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.results_dir)
        self.best_val_bal_acc = -torch.inf
        self.acc_metric = Accuracy().to(self.device)
        self.bal_acc_metric = Accuracy(num_classes=self.n_classes, average="macro").to(self.device)

        # Be sure to store experiment details here for collecting results across runs
        self.metrics = {}

    def __call__(self, epochs: int):
        """Trains the model for a given number of epochs."""
        self.global_step = 0  # batches seen
        self.epoch = 0
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        for _ in trange(epochs, desc="Epochs", leave=False, position=self.tqdm_pos):
            self.writer.add_scalar("epoch", self.epoch, self.global_step)
            self._train()
            if self.val_loader is not None:
                self._val()
            self.epoch += 1
            self.checkpoint()
            self.sched.step()
            self.writer.add_scalar("lr", self.sched.get_last_lr()[0], global_step=self.global_step)

        # After training, load the best model (if available), or else the most recent model
        # NOTE - checked by hand that the *.pt file saved in outer loop is same as best_model.pt,
        # and different than checkpoint.pt
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        try:
            ckpt = torch.load(self.best_ckpt_path)
        except AttributeError:
            ckpt = torch.load(self.ckpt_path)
        self.model.load_state_dict(ckpt["model_state_dict"])

        return self.metrics

    def _run_epoch(self, desc, loader, optim=None):
        self.bal_acc_metric.reset()
        self.acc_metric.reset()
        if optim:
            self.model.train()
        else:
            self.model.eval()

        pbar = tqdm(loader, desc=desc, leave=False, position=self.tqdm_pos + 1)
        total_loss = 0.0
        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            log_probs = self.model(data)
            loss = self.criterion(log_probs, labels)

            if optim:
                optim.zero_grad()
                loss.backward()
                optim.step()

            total_loss += loss.item()
            batch_bal_acc = self.bal_acc_metric(log_probs.argmax(-1), labels)
            batch_acc = self.acc_metric(log_probs.argmax(-1), labels)

            if optim:
                # During training, update metrics each batch
                results = {
                    f"{desc}/batch_loss": float(loss),
                    f"{desc}/batch_acc": batch_acc,
                    f"{desc}/batch_bal_acc": batch_bal_acc,
                }
                pbar.set_postfix({k: f"{v:.3f}" for k, v in results.items()})
                for key, val in results.items():
                    self.writer.add_scalar(key, val, self.global_step)
                self.global_step += 1

        results = {
            f"{desc}/epoch_loss": total_loss / len(loader),
            f"{desc}/epoch_acc": self.acc_metric.compute(),
            f"{desc}/epoch_bal_acc": self.bal_acc_metric.compute(),
        }
        for key, val in results.items():
            self.writer.add_scalar(key, val, self.global_step)
        results["epoch"] = self.epoch
        self.metrics.update(results)

    def _train(self):
        """Training loop"""
        self._run_epoch("train", self.train_loader, self.optim)

    def _val(self):
        self._run_epoch("val", self.val_loader, None)

    def checkpoint(self):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optim_state_dict": self.optim.state_dict(),
        }
        self.ckpt_path = self.results_dir / "checkpoint.pt"
        torch.save(ckpt, self.ckpt_path)
        # TODO - should use "val/epoch_loss" - but there are nans. For now, just use best bal acc.
        # Source of nans is unclear - log_probs from model contain nans in both classes
        # for a single item - but not able to reproduce this by scanning through full train
        # set (including val slice) using model.load("broken.pt") and model.predict(train_x)
        if self.metrics["val/epoch_bal_acc"] > self.best_val_bal_acc:
            self.best_val_bal_acc = self.metrics["val/epoch_bal_acc"]
            self.best_ckpt_path = self.results_dir / "best_model.pt"
            torch.save(ckpt, self.best_ckpt_path)
