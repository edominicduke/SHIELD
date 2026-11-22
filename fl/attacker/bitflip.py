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
from .attacker import Attacker

class BitFlip(Attacker):
    def attack(self, sampled_byz_client_idxs, server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        client_msgs = {}
        for client_idx in sampled_byz_client_idxs:
            client_msg = self.byz_clients[client_idx].local_update(server_message)
            
            update: dict[str, torch.Tensor] = client_msg['update']
            byz_update = {}
            for name in update:
                byz_update[name] = -update[name]
            client_msg['update'] = byz_update
            client_msgs[client_idx] = client_msg

        verbose_log = {}

        return client_msgs, verbose_log
