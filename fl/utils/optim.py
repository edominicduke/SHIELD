"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import SGD
from typing import Iterator

def get_optimizer(optimizer: str, parameters: Iterator[Parameter], lr: float, momentum: float, weight_decay: float):
    if optimizer == 'sgd':
        return SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise Exception('invalid optimizer')

@torch.no_grad()
def step(update: dict[str, torch.Tensor], global_model: nn.Module) -> None:
    model_state: dict[str, torch.Tensor] = global_model.state_dict()
    for name in model_state:
        state_type = model_state[name].dtype
        model_state[name] += update[name].to(dtype=state_type)
    global_model.load_state_dict(model_state)
