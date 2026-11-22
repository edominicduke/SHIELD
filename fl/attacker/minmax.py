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
from .attacker import Attacker

class MinMax(Attacker):
    @torch.no_grad()
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        flat_ref_updates, structure = flatten_updates(ref_updates)
        deviation = MinMax.get_flat_deviation(self.args.minmax_deviation, flat_ref_updates)

        pairwise_dist = torch.cdist(flat_ref_updates, flat_ref_updates)
        max_dist = pairwise_dist.max()


        lamda = 10.0
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        flat_avg = flat_ref_updates.mean(dim=0)
        
        while abs(lamda_succ - lamda) > threshold_diff:
            flat_byz_update = flat_avg - lamda * deviation
            dists: torch.Tensor = torch.norm(flat_byz_update - flat_ref_updates, dim=1)
            dev_dist = dists.max()
            
            if dev_dist <= max_dist:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        flat_byz_update = flat_avg - lamda * deviation
        byz_update = unflatten_update(flat_byz_update, structure)
        sampled_byz_client_msgs = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)

        verbose_log = {'minmax_lamda': lamda}

        return sampled_byz_client_msgs, verbose_log