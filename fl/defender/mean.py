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

from .aggregator import Aggregator
from argparse import Namespace
from typing import Tuple
from ..utils import flatten_updates, unflatten_update

class Mean(Aggregator):
    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        n_datas: list[int] = [client_messages[client_idx]['n_data'] for client_idx in client_messages]
        updates = Aggregator.get_updates(client_messages)

        weight = torch.Tensor(n_datas)
        weight /= weight.sum()

        flat_updates, structure = flatten_updates(updates)
        flat_agg_update = weight.matmul(flat_updates)
        agg_update = unflatten_update(flat_agg_update, structure)

        verbose_log = {}
        
        return agg_update, verbose_log