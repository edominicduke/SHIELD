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
from .aggregator import Aggregator
from ..utils import flatten_updates, unflatten_update

class Aksel(Aggregator):
    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)
        
        flat_median, _ = flat_updates.median(dim=0)
        s = (flat_updates - flat_median).square().sum(dim=-1)
        r = s.median()
        bool_idxs = (s <= r)
        flat_agg_update = flat_updates[bool_idxs].mean(dim=0)
        agg_update = unflatten_update(flat_agg_update, structure)

        candidate_idxs_lt = bool_idxs.nonzero().flatten().tolist()
        cand_info = Aggregator.get_cand_info(structure, candidate_idxs_lt)
        verbose_log = {'cand_info': cand_info}

        return agg_update, verbose_log