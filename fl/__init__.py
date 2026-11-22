"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
from __future__ import annotations

import math
import numpy.random as random
import time
import torch  # * needed for the GPU speed-up switch and the profiling timer
from . import utils

from torch.utils.data.dataloader import DataLoader
from .defender import Defender
from .attacker import get_attacker
from .client import Client

def update_args(args, n_clients: int):
    setattr(args, 'n_clients', n_clients)
    # compute number of byzantine clients
    setattr(args, 'n_byz_clients', math.ceil(args.n_clients * args.byz_client_ratio))

    # sychronize n_sampled_clients and client_sample_ratio
    if args.n_sampled_clients is None:
        setattr(args, 'n_sampled_clients', math.ceil(args.n_clients * args.client_sample_ratio))
    else:
        setattr(args, 'client_sample_ratio', args.n_sampled_clients / args.n_clients)

    return args

def _tick(args):  # * profiling timer
    # * wall-clock timestamp for --profile; it first waits for queued GPU work so the phase timings are accurate
    if args.profile and torch.cuda.is_available():  # * only when profiling on a GPU
        torch.cuda.synchronize()  # * wait for queued GPU work
    return time.time()  # * wall-clock seconds


def fl_train(args):
    utils.setup_seed(args.seed)
    # * GPU speed-up switch: silently fall back to the normal path when no CUDA device is present
    if args.gpu_fast and not torch.cuda.is_available():  # * requested but unavailable
        print('--gpu_fast requested but no CUDA device found: running the normal path')  # * tell the user about the fallback
        args.gpu_fast = False  # * fall back to the normal path
    utils.set_fast(args.gpu_fast)  # * tells the statistics functions in utils to use the GPU
    # load data
    client_datas, te_data = utils.load_data(args.dataset, args.data_root, args.n_clients, args.partition, args.dirichlet_beta, args.dirichlet_min_n_data)
    update_args(args, len(client_datas))
    # record arguments
    print(f'args: {args}')
    # initialize client-side
    clients = {str(idx): Client(client_datas[idx], args) for idx in range(len(client_datas))}
    # initialize attacker
    byz_client_idxs = {str(idx) for idx in range(args.n_byz_clients)}
    byz_clients = {idx: clients[idx] for idx in byz_client_idxs}
    attacker = get_attacker(byz_clients, args=args)
    # initialize server-side
    global_model = utils.get_model(args.architecture, args.n_classes)
    defender = Defender(args, byz_client_idxs)
    # test
    te_dataloader = None  # * built below
    if args.gpu_fast:  # * GPU-resident test set (None for unsupported datasets such as ImageNet12)
        te_dataloader = utils.make_gpu_loader(te_data, args.te_batch_size, shuffle=False)  # * GPU-resident test loader
    if te_dataloader is None:  # * unsupported dataset or fast path off: normal DataLoader
        te_dataloader = DataLoader(dataset=te_data, batch_size=args.te_batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=args.pin_memory)
    loss_fun = utils.get_loss(args.loss)

    # start
    for epoch in range(args.server_n_epochs):
        tic = time.time()
        timing = {}  # * per-phase seconds for --profile
        # sample clients
        sampled_client_idxs: set[str] = set(str(idx) for idx in random.choice(args.n_clients, args.n_sampled_clients, replace=False).tolist())
        # distribute
        server_message = {'model_state': global_model.state_dict()}
        # benign clients perform local update
        sampled_ben_client_idxs = sampled_client_idxs.difference(byz_client_idxs)
        t_a = _tick(args)  # * profile: start of local training
        ben_client_msgs = {idx: clients[idx].local_update(server_message) for idx in sampled_ben_client_idxs}
        timing['t_local_train'] = _tick(args) - t_a  # * profile: local training of the benign clients
        # attack
        num_sampled_bex = len(ben_client_msgs)
        num_known_benign = int(args.ben_known_ratio * num_sampled_bex)
        known_benign_subidx = random.choice(num_sampled_bex, num_known_benign, replace=False).tolist()
        sampled_ben_client_idxs_lt = list(sampled_ben_client_idxs)
        known_benign_client_idxs = [sampled_ben_client_idxs_lt[i] for i in known_benign_subidx]
        sampled_byz_client_idxs = sampled_client_idxs.intersection(byz_client_idxs)
        # * blind defenses: instead of the true Byzantine count, every defense (and the oracle queries of the attacker)
        # * receives an ASSUMED count = floor(ratio * n_sampled). By default the ratio equals SHIELD's rejection cap,
        # * so all defenses are given the same upper bound on the attacker fraction. The true count is still used
        # * for the attack itself and for evaluation metrics
        if args.blind_defenses:  # * opt-in; off keeps the starter-code behavior
            assumed_ratio = args.assumed_byz_ratio if args.assumed_byz_ratio is not None else args.shield_max_reject_frac  # * default = SHIELD's cap
            defender_knowledge = {'n_byz_updates': int(math.floor(assumed_ratio * len(sampled_client_idxs)))}  # * assumed, not true, count
        else:  # * starter-code behavior
            defender_knowledge = {'n_byz_updates': len(sampled_byz_client_idxs)}  # * starter-code behavior: true count
        attacker_knowledge = {
            'benign_client_messages': ben_client_msgs,
            'known_benign_client_idxs': known_benign_client_idxs,
            'defender': defender, 
            'defender_knowledge': defender_knowledge, 
        }
        t_a = _tick(args)  # * profile: start of the attack
        byz_client_msgs, byz_verbose_log = attacker.attack(sampled_byz_client_idxs, server_message, attacker_knowledge)
        timing['t_attack'] = _tick(args) - t_a  # * profile: attack (including any oracle queries)
        # mergetorch slice
        sampled_client_msgs = {**ben_client_msgs, **byz_client_msgs}
        # aggregate
        t_a = _tick(args)  # * profile: start of the defense
        agg_update, agg_verbose_log = defender.defend(sampled_client_msgs, defender_knowledge)
        timing['t_defend'] = _tick(args) - t_a  # * profile: SHIELD / aggregator
        # step
        utils.step(agg_update, global_model)
        # evaluate
        t_a = _tick(args)  # * profile: start of the evaluation statistics
        ben_stats = utils.eval_ben_updates(ben_client_msgs)
        byz_dev_stats = utils.eval_byz_dev(byz_client_msgs, ben_client_msgs)
        agg_dev_stats = utils.eval_agg_dev(agg_update, ben_client_msgs)
        perf_stats = utils.eval_perf(global_model, te_dataloader, loss_fun)
        timing['t_eval'] = _tick(args) - t_a  # * profile: PCA statistics, deviations and test-set evaluation
        # * pipeline-level selection metrics (precision, recall, benign FPR of rejected updates), computed only for
        # * defenses that select or reject updates: Krum/Bulyan/Aksel/DnC/MDA report their kept candidates, and
        # * SHIELD in active mode reports what it rejected. Shadow mode is skipped (its training path is the oracle)
        sel_stats = {}  # * empty unless the defense has a selection step
        cand_info = agg_verbose_log.get('cand_info') or agg_verbose_log.get('krum_cand_info')  # * kept-candidate info
        if args.shield != 'shadow':  # * shadow mode trains on the oracle path, so skip
            if cand_info is not None:  # * aggregator reported its kept candidates
                kept_ids = set(cand_info['cand_client_idxs_lt'])  # * ids the aggregator kept (bucket ids are split)
                sel_stats = utils.eval_selection(kept_ids, sampled_client_idxs, byz_client_idxs)  # * metrics of the rejected updates
            elif args.shield == 'active':  # * SHIELD active without a candidate list
                kept_ids = set(sampled_client_idxs) - defender.shield_rejected  # * everything SHIELD did not reject
                sel_stats = utils.eval_selection(kept_ids, sampled_client_idxs, byz_client_idxs)  # * metrics of the rejected updates
        # * SHIELD Phase 0: optional diagnostics (empty dict when disabled, so logging is unchanged)
        phase0_stats = utils.eval_phase0_diagnostics(ben_client_msgs, byz_client_msgs) if args.phase0_diag else {}
        # wandb log
        if args.wandb:
            import wandb
            log_data = {
                'sampled_client_idxs': list(sampled_client_idxs),
                **ben_stats,
                **byz_verbose_log,
                **byz_dev_stats,
                **agg_verbose_log,
                **agg_dev_stats,
                **perf_stats,
                **phase0_stats,  # * SHIELD Phase 0 diagnostics
                **defender.shield_log,  # * SHIELD per-round log
                **sel_stats,  # * selection metrics
            }
            wandb.log(log_data) # type: ignore
        if not args.nice and epoch == 0:
            utils.block(args.devices)
        toc = time.time()
        # output
        print(f'==================== epoch {epoch} time {toc - tic: .2f} ====================')
        print(f'sampled clients: {sampled_client_idxs}')
        utils.print_dict(ben_stats, 'ben_updates stats')
        utils.print_dict(byz_verbose_log, args.attack)
        utils.print_dict(byz_dev_stats, 'byz_update dev stats')
        utils.print_dict(agg_verbose_log, args.aggregator)
        utils.print_dict(agg_dev_stats, 'agg_update dev stats')
        utils.print_dict(perf_stats, f'performance {epoch}')
        utils.print_dict(phase0_stats, 'SHIELD phase0 diagnostics')  # * prints nothing when disabled
        if args.profile:  # * per-phase timing, parsed by the evaluation notebook
            utils.print_dict({**timing, 't_round': toc - tic}, 'timing (s)')  # * print the per-phase timings
        utils.print_dict(sel_stats, 'selection metrics')  # * prints nothing for defenses without a selection step
        utils.print_dict(defender.shield_log, 'SHIELD')  # * SHIELD per-round log; prints nothing when SHIELD is off
        

    print('==================== end of training ====================')

__all__ = [
    'fl_train',
]