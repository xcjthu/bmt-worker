import torch.optim as optim
import torch
from torch.optim import AdamW
import bmtrain as bmt


def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("train", "optimizer")
    learning_rate = config.getfloat("train", "learning_rate")

    param_group = model.parameters()


    optimizer = bmt.optim.AdamOffloadOptimizer(param_group, lr=learning_rate,
                                weight_decay=config.getfloat("train", "weight_decay"))#, scale=2**20)
    # if optimizer_type == "adam":
    #     optimizer = optim.Adam(param_group, lr=learning_rate,
    #                            weight_decay=config.getfloat("train", "weight_decay"))
    # elif optimizer_type == "sgd":
    #     optimizer = optim.SGD(param_group, lr=learning_rate,
    #                           weight_decay=config.getfloat("train", "weight_decay"))
    # elif optimizer_type == "AdamW":
    #     optimizer = AdamW(param_group, lr=learning_rate,
    #                          weight_decay=config.getfloat("train", "weight_decay"))
    # else:
    #     raise NotImplementedError

    return optimizer
