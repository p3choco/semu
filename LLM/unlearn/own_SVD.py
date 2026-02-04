import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .own.impl import iterative_unlearn
from .own.utils import train_phase


class OwnSVD:
    @staticmethod
    def train_iter(data_loaders, model, criterion, optimizer, epoch, args):
        forget_loader = data_loaders["forget_train"]

        losses = utils.AverageMeter()

        # switch to train mode
        train_phase(model, changed_layers_class=["customlinear", "customconv2d"])

        train_dataset = deepcopy(forget_loader.dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        start = time.time()
        for i, prompt in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

            loss = criterion(model, prompt)

            assert (
                torch.isfinite(loss).all().item()
            ), f"Loss is NaN or Infinite, get: {loss}"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.float()

            #  record loss
            losses.update(loss.item(), prompt['input_ids'].size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses
                    )
                )
                start = time.time()


@iterative_unlearn
def own_svd():
    return OwnSVD
