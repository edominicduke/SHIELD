"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
from __future__ import annotations

from argparse import Namespace

from .bucket import bucket
from .nnm import nnm

def get_aggregator(args: Namespace, byz_client_idxs: set[str]):
    if args.aggregator == 'aksel':
        from .aksel import Aksel
        return Aksel(args, byz_client_idxs)
    elif args.aggregator == 'bulyan':
        from .bulyan import Bulyan
        return Bulyan(args, byz_client_idxs)
    elif args.aggregator == 'cc':
        from .centeredclipping import CenteredClipping
        return CenteredClipping(args, byz_client_idxs)
    elif args.aggregator == 'clamp':
        from .clamp import Clamp
        return Clamp(args, byz_client_idxs)
    elif args.aggregator == 'dnc':
        from .dnc import Dnc
        return Dnc(args, byz_client_idxs)
    elif args.aggregator == 'krum':
        from .krum import Krum
        return Krum(args, byz_client_idxs)
    elif args.aggregator == 'mda':
        from .mda import Mda
        return Mda(args, byz_client_idxs)
    elif args.aggregator == 'mean':
        from .mean import Mean
        return Mean(args, byz_client_idxs)
    elif args.aggregator == 'median':
        from .median import Median
        return Median(args, byz_client_idxs)
    elif args.aggregator == 'rfa':
        from .rfa import Rfa
        return Rfa(args, byz_client_idxs)
    elif args.aggregator == 'trmean':
        from .trmean import Trmean
        return Trmean(args, byz_client_idxs)
    else:
        raise Exception('invalid aggregator')

class Defender:
    def __init__(self, args: Namespace, byz_client_idxs: set[str]) -> None:
        self.args = args
        self.aggregator = get_aggregator(args, byz_client_idxs)
        # * SHIELD: optional pre-filter ('off', 'shadow' or 'active'); the import is lazy so 'off' costs nothing
        self.byz_client_idxs = byz_client_idxs  # * kept so shadow mode can build the oracle clean path
        self.shield_mode = getattr(args, 'shield', 'off')  # * 'off' when the flag is absent
        self.shield = None  # * no SHIELD object unless enabled
        if self.shield_mode != 'off':  # * only build SHIELD when requested
            from .shield import Shield  # * lazy import
            self.shield = Shield(args, byz_client_idxs)  # * stateful across rounds
        # * most recent committed SHIELD log, read by fl_train for printing and wandb
        self.shield_log: dict = {}  # * empty until SHIELD commits a round
        self.shield_rejected: set = set()  # * ids SHIELD rejected in the latest committed round (active mode)

    def defend(self, client_msgs: dict[str, dict], knowledge: dict, commit: bool = True):  # * added commit flag for SHIELD
        # * commit=False is used for attacker oracle queries (omn), so they cannot change SHIELD's state
        if self.shield is not None:  # * SHIELD runs before nnm/bucket/aggregator
            accepted_ids, shield_log = self.shield.process(client_msgs, commit=commit)  # * score, log, pick accepted ids
            if commit:  # * only real rounds update the log
                self.shield_log = shield_log  # * store for printing and wandb
                self.shield_rejected = set(client_msgs) - set(accepted_ids)  # * ids SHIELD rejected this round
            if self.shield_mode == 'shadow':  # * oracle clean path, see comment below
                # * shadow mode, oracle clean path: SHIELD only scores and logs, while training aggregates the
                # * benign updates only (simulator ground truth that SHIELD itself never sees)
                client_msgs = {i: m for i, m in client_msgs.items() if i not in self.byz_client_idxs}  # * drop Byzantine updates (ground truth)
            else:
                # * active mode: hard rejection, the downstream pipeline sees only the accepted updates
                client_msgs = {i: client_msgs[i] for i in accepted_ids}  # * keep only accepted updates
            # * SHIELD is blind to the true Byzantine count, so the downstream aggregator is told 0
            knowledge = {**knowledge, 'n_byz_updates': 0}  # * copy, do not mutate the caller's dict
        if self.args.nnm:
            client_msgs, knowledge = nnm(client_msgs, knowledge)
        bucket_client_msgs, bucket_knowledge = bucket(client_msgs, self.args.bucket_s, knowledge)
        agg_update, verbose_log = self.aggregator(bucket_client_msgs, bucket_knowledge)
        return agg_update, verbose_log