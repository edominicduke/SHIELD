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

from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .aggregator import Aggregator

class Trmean(Aggregator):
    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)
        n_updates = len(updates)
        n_byz_updates = min(knowledge['n_byz_updates'], (n_updates - 1) // 2)
        if n_byz_updates == 0:
            flat_agg_update = flat_updates.mean(dim=0)
        else:
            sorted_flat_updates , _ = flat_updates.sort(dim=0)
            flat_agg_update = sorted_flat_updates[n_byz_updates:-n_byz_updates].mean(dim=0)
        
        agg_update = unflatten_update(flat_agg_update, structure)

        verbose_log = {}

        return agg_update, verbose_log