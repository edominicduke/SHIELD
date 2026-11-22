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

from argparse import Namespace
from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .aggregator import Aggregator

class CenteredClipping(Aggregator):
    def __init__(self, args: Namespace, byz_client_idxs: set[str]) -> None:
        super().__init__(args, byz_client_idxs)
        self.flat_agg_update = None

    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)
        # initialize v
        if self.flat_agg_update is None:
            d = flat_updates.size(-1)
            v = torch.zeros(d)
        else:
            v = self.flat_agg_update
        # update v
        all_client_weights = []
        for _ in range(self.args.cc_l):
            diffs = flat_updates - v
            weights: torch.Tensor = self.args.cc_tau / diffs.norm(dim=-1)
            weights = weights.clamp_max(max=1)
            c = diffs * weights.view(-1, 1)
            v += c.mean(dim=0)

            client_weights = {idx: weight for idx, weight in zip(structure['client_idxs'], weights.tolist())}
            all_client_weights.append(client_weights)
        self.flat_agg_update = v
        agg_update = unflatten_update(v, structure)

        verbose_log = {'all_client_weights': all_client_weights}

        return agg_update, verbose_log