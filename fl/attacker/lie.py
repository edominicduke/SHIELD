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

class Lie(Attacker):
    @torch.no_grad()
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        flat_ref_updates, structure = flatten_updates(ref_updates)

        flat_avg = flat_ref_updates.mean(dim=0)
        flat_std = flat_ref_updates.std(dim=0, unbiased=False)

        flat_byz_update = flat_avg - self.args.lie_z * flat_std
        byz_update = unflatten_update(flat_byz_update, structure)

        byz_client_msgs = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)

        verbose_log = {}

        return byz_client_msgs, verbose_log
            
