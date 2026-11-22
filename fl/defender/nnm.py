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
from .. import utils

def nnm(client_msgs: dict[str, dict], knowledge: dict):
    # flatten
    n_clients = len(client_msgs)
    n_byz = knowledge['n_byz_updates']
    updates = {idx: client_msg['update'] for idx, client_msg in client_msgs.items()}
    flat_updates, structure = utils.flatten_updates(updates)
    # compute distance
    dists = torch.cdist(flat_updates, flat_updates)
    # find nearest neighbors
    nearest_neighbor_idxs = dists.topk(k=n_clients - n_byz + 1, largest=False, sorted=True)[1][..., 1:]
    # mix
    mixed_cl_messages = {}
    for idx, nearest_neighbor_idxs_per_cl in zip(client_msgs, nearest_neighbor_idxs):
        flat_mixed_update = flat_updates[nearest_neighbor_idxs_per_cl].mean(dim=0)
        mixed_update = utils.unflatten_update(flat_mixed_update, structure)
        mix_cl_message = {
                'update': mixed_update,
                'n_data': client_msgs[idx]['n_data'],
        }
        mixed_cl_messages[idx] = mix_cl_message
    return mixed_cl_messages, knowledge