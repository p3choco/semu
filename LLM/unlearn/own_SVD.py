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
        
        print("Hello form train_iter!")
        forget_loader = data_loaders["forget"]
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode: model.train()

        mode = "EVAL"
        if model.training:
            mode = "TRAIN"
        print(f"Model mode: {mode}")

        # forget_dataset = deepcopy(forget_loader.dataset)
        train_dataset = deepcopy(forget_loader.dataset)

        # train_phase(model, changed_layers_class=["customlinear", "customconv2d"])

        # if hasattr(forget_dataset, 'targets'):
        #     train_dataset = forget_dataset
        # else:
        #     train_dataset = forget_dataset.dataset

        # targets_unlearn = np.random.randint(0, args.num_classes, len(train_dataset.targets))
        # targets_unlearn = np.where(
        #     targets_unlearn == train_dataset.targets,
        #     np.remainder(targets_unlearn + 1, args.num_classes),
        #     targets_unlearn
        # )
        # train_dataset.targets = targets_unlearn

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # model.train()
        # model.enable_input_require_grads()

        sum_gradients = {}

        for i, batch in enumerate(train_loader):
            loss, _ = criterion(model, batch)

            assert torch.isfinite(loss).all().item(), \
                f"Loss is NaN or Infinite, got: {loss}"

            model.zero_grad(set_to_none=True)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in sum_gradients:
                        sum_gradients[name] = param.grad.detach().cpu().clone()
                    else:
                        sum_gradients[name] += param.grad.detach().cpu()

            del loss
            torch.cuda.empty_cache()


        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # start = time.time()
        # for i, prompt in enumerate(train_loader):
        #     if epoch < args.warmup:
        #         utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

        #     # image = image.cuda()
        #     # target = target.cuda()

        #     output_clean = model(prompt)
        #     loss, _ = criterion(model, prompt)

        #     assert (
        #         torch.isfinite(loss).all().item()
        #     ), f"Loss is NaN or Infinite, get: {loss}"

        #     print("Czy tu się wywalam? 1")
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     print("Jednak NIE...")

        #     print("Czy tu się wywalam? 2")
        #     output = output_clean.float()
        #     loss = loss.float()
        #     prec1 = utils.accuracy(output.data, target)[0]
        #     print("Jednak NIE...")

        #     print("Czy tu się wywalam 3?")
        #     losses.update(loss.item(), image.size(0))
        #     top1.update(prec1.item(), image.size(0))
        #     print("Jednak NIE...")

        #     if (i + 1) % args.print_freq == 0:
        #         end = time.time()
        #         print(
        #             "Epoch: [{0}][{1}/{2}]\t"
        #             "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
        #             "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
        #             "Time {3:.2f}".format(
        #                 epoch, i, len(train_loader), end - start, loss=losses, top1=top1
        #             )
        #         )
        #         start = time.time()

        # print("train_accuracy {top1.avg:.3f}".format(top1=top1))

        # return top1.avg

@iterative_unlearn
def own_svd():
    return OwnSVD
