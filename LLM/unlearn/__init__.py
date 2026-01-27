from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint

from .own_SVD import own_svd


def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "own_SVD":
        return own_svd
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")