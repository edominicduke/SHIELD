# * New file (not from the starter code): SHIELD-aware STRIKE ('skew_aware').
# * It starts from the ordinary STRIKE update and then adapts it to evade SHIELD in up to three independent ways:
# *   1. aware_project: remove the part of STRIKE's deviation that lies OUTSIDE the span of the benign updates
# *      (the component SHIELD's residual feature looks at), keeping only the in-span part
# *   2. aware_noise: give every Byzantine client its own in-span noise so the forged copies are not identical
# *      (breaks the duplicate-cluster tell)
# *   3. aware_oracle: query the defender (accept / reject only, like Omn does) and bisection-search the largest
# *      deviation strength that still gets all forged updates accepted
# * The attacker is a black-box oracle user: it only observes which of its updates were accepted. It never reads
# * SHIELD's private random generator or its ground-truth bookkeeping.
from __future__ import annotations

import torch

from argparse import Namespace
from typing import Tuple
from ..client import Client
from ..utils import flatten_update, flatten_updates, unflatten_update
from .skew import Skew


class SkewAware(Skew):
    def __init__(self, byz_clients: dict[str, Client], args: Namespace) -> None:
        super().__init__(byz_clients, args)
        # * private torch generator for the attacker's noise, so it never touches the global training RNG
        self._gen = torch.Generator().manual_seed(int(args.seed) * 7919 + 17)

    @torch.no_grad()
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        # * step 1: the ordinary STRIKE update (also gives the log entries of the base attack)
        base_msgs, log = super().attack(sampled_byz_client_idxs, server_message, knowledge)

        # * same reference updates as STRIKE itself uses (known benign subset, else all benign updates)
        if len(knowledge['known_benign_client_idxs']) == 0:
            ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        else:
            ref_updates = {idx: knowledge['benign_client_messages'][idx]['update'] for idx in knowledge['known_benign_client_idxs']}
        flat_ref, structure = flatten_updates(ref_updates)  # * [n_ref x n_params]
        n_ref = flat_ref.size(0)
        mean_ref = flat_ref.mean(dim=0)  # * center of the reference updates

        # * STRIKE's forged update and its deviation from the reference mean
        flat_strike, _ = flatten_update(next(iter(base_msgs.values()))['update'])
        dev = flat_strike - mean_ref
        dev_norm = dev.norm().item()

        # * centered reference updates, whose row space is the benign span
        centered = flat_ref - mean_ref

        # * step 2 (aware_project): keep only the part of the deviation inside the benign span
        if self.args.aware_project and n_ref >= 2 and dev_norm > 0:
            gram = (centered @ centered.T).double()  # * n_ref x n_ref
            coef = torch.linalg.pinv(gram, hermitian=True) @ (centered @ dev).double()  # * least-squares coefficients
            dev_dir = centered.T @ coef.float()  # * projection of dev onto the benign span
            proj_norm = dev_dir.norm().item()
            # * the largest scale that restores STRIKE's original deviation magnitude inside the span
            t_max = dev_norm / (proj_norm + 1e-12)
            proj_ratio = proj_norm / dev_norm
        else:
            dev_dir, t_max, proj_ratio = dev, 1.0, 1.0  # * no projection: scale 1 is STRIKE's own strength

        # * step 3 (aware_noise): independent in-span noise per Byzantine client (zero noise = identical copies)
        byz_ids = sorted(sampled_byz_client_idxs)
        noises = {}
        for idx in byz_ids:
            if self.args.aware_noise > 0 and n_ref >= 2:
                w = torch.randn(n_ref, generator=self._gen)  # * random combination of the centered benign updates
                noises[idx] = self.args.aware_noise * (centered.T @ w) / (n_ref ** 0.5)
            else:
                noises[idx] = torch.zeros_like(mean_ref)

        def build_msgs(t: float) -> dict[str, dict]:
            # * forged messages at strength t: reference mean + t * (in-span deviation) + this client's noise
            msgs = {}
            for idx in byz_ids:
                msg = self.byz_clients[idx].pack_other_message()  # * same metadata as an honest message
                msg['update'] = unflatten_update(mean_ref + t * dev_dir + noises[idx], structure)
                msgs[idx] = msg
            return msgs

        # * step 4 (aware_oracle): black-box queries of the defender, accept / reject only
        defender = knowledge['defender']
        benign_msgs = knowledge['benign_client_messages']
        n_queries = 0
        t = t_max
        if self.args.aware_oracle and getattr(defender, 'shield', None) is not None and len(benign_msgs) > 0:
            def accepted(strength: float) -> bool:
                # * True if SHIELD (queried without committing any state) accepts every forged update
                forged = build_msgs(strength)
                acc_ids, _ = defender.shield.process({**benign_msgs, **forged}, commit=False)
                return all(idx in set(acc_ids) for idx in forged)

            n_queries += 1
            if not accepted(t_max):
                lo, hi = 0.0, t_max  # * largest accepted strength lies in [lo, hi]
                for _ in range(self.args.aware_queries):
                    mid = (lo + hi) / 2
                    n_queries += 1
                    if accepted(mid):
                        lo = mid
                    else:
                        hi = mid
                t = lo  # * strongest strength that was accepted (0 = nothing but the mean got through)

        # * final forged messages at the chosen strength
        log = {**log, 'aware_t': t, 'aware_t_max': t_max, 'aware_proj_ratio': proj_ratio, 'aware_queries': n_queries}
        return build_msgs(t), log
