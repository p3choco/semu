import finetune

from . import utils


def synflow(model, train_loader, test_loader, criterion, args):
    optimizer, scheduler = finetune.get_optimizer_and_scheduler(model, args)

    if args.rate != 0:
        utils.global_prune_model(model, args.rate, "synflow", train_loader)
        utils.check_sparsity(model)

    finetune.train_with_rewind(
        model, optimizer, scheduler, train_loader, criterion, args
    )
    utils.check_sparsity(model)