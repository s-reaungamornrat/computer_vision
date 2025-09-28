import torch
import torch.nn as nn

def intersect_dicts(da, db, exclude=()):
    """
    Return a dict of intersecting keys with matching shapes, excluding `exclude` keys
    Args:
        da (dict): First dict
        db (dict): Second dict
        exclude (tuple, optional): Keys to exclude
    Returns:
        (dict): Dict of intersecting keys with matching shapes
    """
    return {k:v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape==db[k].shape}

def initialize_weights(model):
    """
    Initialize model weights to random values
    """
    for m in model.modules():
        t=type(m)
        if t is nn.BatchNorm2d:
            m.eps=1.e-3
            m.momentum=0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.implace=True
