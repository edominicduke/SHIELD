"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
from __future__ import annotations

import itertools
import torch

from argparse import Namespace
from typing import Iterable, Tuple

class Aggregator:
    byz_client_idxs = set()

    def __init__(self, args: Namespace, byz_client_idxs: set[str]) -> None:
        self.args = args
        Aggregator.byz_client_idxs = byz_client_idxs

    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        raise Exception('specification of aggregator is needed')

    @staticmethod
    def get_updates(client_messages: dict) -> dict[str, dict[str, torch.Tensor]]:
        updates: dict[str, dict[str, torch.Tensor]] = {client_idx: client_messages[client_idx]['update'] for client_idx in client_messages}
        return updates
    
    @staticmethod
    def get_cand_info(structure: dict, candidate_idxs: Iterable[int]) -> dict:
        sampled_client_idxs = structure['client_idxs']
        cand_client_idxs_lt = list(itertools.chain.from_iterable(sampled_client_idxs[idx].split('_') for idx in candidate_idxs))
        n_cand_byz_clients = len(Aggregator.byz_client_idxs.intersection(cand_client_idxs_lt))
        n_cand_ben_clients = len(cand_client_idxs_lt) - n_cand_byz_clients
        cand_info = {
            'cand_client_idxs_lt': cand_client_idxs_lt,
            'n_cand_byz_clients': n_cand_byz_clients,
            'n_cand_ben_clients': n_cand_ben_clients,
        }
        return cand_info