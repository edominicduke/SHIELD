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
import torch.nn.functional as F

from typing import Tuple

# * GPU speed-up switch (new, not part of the starter code), set once by fl_train from --gpu_fast. When on, the
# * heavy [n_clients x n_params] statistics below are computed on the GPU instead of the CPU
_FAST = False  # * default: CPU statistics


def set_fast(flag) -> None:  # * switch setter
    # * turn the GPU statistics path on or off
    global _FAST  # * module-level flag
    _FAST = bool(flag)  # * store the flag


def _maybe_gpu(t: torch.Tensor) -> torch.Tensor:  # * move a tensor to the GPU when enabled
    # * move a tensor to the GPU only when the speed-up is on and a GPU exists
    return t.cuda() if (_FAST and torch.cuda.is_available()) else t  # * no-op when disabled or no GPU

@torch.no_grad()
def flatten_update(update: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, list[Tuple[str, torch.Size]]]:
    flat_param_updates = []
    structure = []
    for name, param_update in update.items():
        flat_param_updates.append(param_update.view(-1))
        structure.append((name, param_update.size()))
    flat_update = torch.cat(flat_param_updates)

    return flat_update, structure

@torch.no_grad()
def flatten_updates(updates: dict[str, dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, dict]:
    flat_updates = []
    client_idxs = []
    update_structure = []
    for client_idx in updates:
        client_idxs.append(client_idx)
        update = updates[client_idx]
        flat_update, update_structure = flatten_update(update)
        flat_updates.append(flat_update)
    if len(flat_updates) == 0:
        flat_updates = torch.Tensor()
    else:
        flat_updates = torch.stack(flat_updates)
    structure = {
        'client_idxs': client_idxs,
        'update_structure': update_structure,
    }
    return flat_updates, structure

@torch.no_grad()
def unflatten_update(flat_update: torch.Tensor, structure: dict) -> dict[str, torch.Tensor]:
    update_structure: list[Tuple[str, torch.Size]] = structure['update_structure']
    update = {}
    tensor_idx = 0
    for name, size in update_structure:
        num = size.numel()
        update[name] = flat_update[tensor_idx : tensor_idx + num].view(size)
        tensor_idx += num
    return update

@torch.no_grad()
def eval_ben_updates(benign_client_messages: dict[str, dict], eps: float=1e-7) -> dict:
    if len(benign_client_messages) == 0:
        return {}
    else:
        benign_updates = {idx: benign_client_messages[idx]['update'] for idx in benign_client_messages}
        flat_benign_updates, _ = flatten_updates(benign_updates)
        flat_benign_updates = _maybe_gpu(flat_benign_updates)  # * GPU path: the PCA below is the heaviest CPU step
        mean = flat_benign_updates.mean(dim=0)
        # dimension-wise
        
        # principle compoment
        q = len(flat_benign_updates)
        U, S, V = torch.pca_lowrank(flat_benign_updates, q=q, center=True)
        principle_coordinates = U[:, 0]
        pca_mean = principle_coordinates.mean().item()
        # pca_median = principle_coordinates.median().item()
        # pca_median_shift = pca_median - pca_mean
        pca_std = principle_coordinates.std(unbiased=False).item()
        pca_moment3 = (principle_coordinates - pca_mean).pow(3).mean().item()
        pca_fisher = pca_moment3 / (pca_std ** 3 + eps)

        pca_dir = V[:, 0]
        ipm_dir = -mean
        pca_vs_ipm = F.cosine_similarity(pca_dir, ipm_dir, dim=0).item()
        skew_pca_dir = pca_fisher * pca_dir
        skew_pca_vs_ipm = F.cosine_similarity(skew_pca_dir, ipm_dir, dim=0).item()
        lie_dir = flat_benign_updates.std(dim=0)
        pca_vs_lie = F.cosine_similarity(pca_dir, lie_dir, dim=0).item()
        

        ben_stats = {
            'principle_coordinates': principle_coordinates.tolist(),

            'benign_avg_norm': mean.norm().item(),
            'singular_values': S.tolist(),
            # 'pca_std': pca_std,

            'pca_vs_ipm': pca_vs_ipm, 
            'skew_pca_vs_ipm': skew_pca_vs_ipm, 
            'pca_vs_lie': pca_vs_lie, 

            'pca_fisher': pca_fisher,
            # 'pca_median_shift': pca_median_shift,
        }

        return ben_stats

def eval_agg_dev(agg_update: dict[str, torch.Tensor], ben_client_msgs: dict[str, dict]):
    if len(ben_client_msgs) == 0:
        return {}
    else:
        flat_agg_update, _ = flatten_update(agg_update)

        benign_updates = {idx: ben_client_msgs[idx]['update'] for idx in ben_client_msgs}
        flat_benign_updates, _ = flatten_updates(benign_updates)
        flat_benign_avg = flat_benign_updates.mean(dim=0)

        renamed_dev_stats = {}
        dev_stats = eval_dev(flat_agg_update, flat_benign_avg)
        for key, val in dev_stats.items():
            renamed_dev_stats['agg_'+key] = val
        
        return renamed_dev_stats

def eval_byz_dev(byz_client_msgs: dict[str, dict], ben_client_msgs: dict[str, dict]):
    if len(byz_client_msgs) == 0 or len(ben_client_msgs) == 0:
        return {}
    else:
        # flat_byz_avg
        byz_updates = {idx: msg['update'] for idx, msg in byz_client_msgs.items()}
        flat_byz_updates, _ = flatten_updates(byz_updates)
        flat_byz_avg = flat_byz_updates.mean(dim=0)
        # flat_ben_avg
        ben_updates = {idx: msg['update'] for idx, msg in ben_client_msgs.items()}
        flat_ben_updates, _ = flatten_updates(ben_updates)
        flat_ben_avg = flat_ben_updates.mean(dim=0)

        renamed_dev_stats = {}
        dev_stats = eval_dev(flat_byz_avg, flat_ben_avg)
        for key, val in dev_stats.items():
            renamed_dev_stats['byz_'+key] = val
        
        return renamed_dev_stats

def eval_dev(target_update: torch.Tensor, ref_update: torch.Tensor, eps: float=1e-7):
    dev = target_update - ref_update

    dev_norm: float = dev.norm().item()
    ref_norm: float = ref_update.norm().item()
    cosine = dev.dot(ref_update).item() / (dev_norm + eps) / (ref_norm + eps)

    dev_stats = {
        'dev_cos': cosine,
        'dev_norm': dev_norm,
        # 'ref_norm': ref_norm,
    }

    return dev_stats

# * ---------------------------------------------------------------------------
# * SHIELD Phase 0 diagnostics (new, not part of the starter code).
# * Measures, per round, whether Byzantine updates differ from benign updates in
# * a way that is invisible in low dimensions but visible in higher ones.
# * ---------------------------------------------------------------------------
@torch.no_grad()
def eval_phase0_diagnostics(
    ben_client_msgs: dict[str, dict],
    byz_client_msgs: dict[str, dict],
    ks: Tuple[int, ...] = (2, 5, 10, 20),
    eps: float = 1e-7,
) -> dict:
    # * need at least 3 benign updates (for a meaningful subspace) and 1 Byzantine update
    if len(ben_client_msgs) < 3 or len(byz_client_msgs) == 0:
        return {}

    # * flatten benign and Byzantine updates into [n_clients x n_params] matrices
    flat_ben, _ = flatten_updates({idx: msg['update'] for idx, msg in ben_client_msgs.items()})
    flat_byz, _ = flatten_updates({idx: msg['update'] for idx, msg in byz_client_msgs.items()})
    flat_ben, flat_byz = _maybe_gpu(flat_ben), _maybe_gpu(flat_byz)  # * GPU path for the Gram / distance computations
    n_ben = flat_ben.size(0)

    stats = {}

    # * (a) pairwise-distance contrast among benign updates: (max - min) / min over distinct pairs.
    # * near 0 => all pairs look equally far apart (distance concentration); large => a meaningful "crowd"
    ben_dists = torch.cdist(flat_ben, flat_ben)
    off_diag_mask = ~torch.eye(n_ben, dtype=torch.bool, device=flat_ben.device)  # * same device as the data
    off_diag = ben_dists[off_diag_mask]
    stats['p0_ben_dist_contrast'] = ((off_diag.max() - off_diag.min()) / (off_diag.min() + eps)).item()
    # * mean benign pairwise distance, used as a scale reference
    stats['p0_ben_mean_pairwise_dist'] = off_diag.mean().item()

    # * how far each Byzantine update is from its nearest benign update, relative to how far
    # * benign updates are from their own nearest benign neighbor. a ratio <= 1 means the
    # * Byzantine update sits inside the crowd by full-dimensional L2 distance
    ben_nn_dist = ben_dists.masked_fill(~off_diag_mask, float('inf')).min(dim=1)[0]
    byz_to_ben_nn_dist = torch.cdist(flat_byz, flat_ben).min(dim=1)[0]
    stats['p0_byz_nn_dist_ratio'] = (byz_to_ben_nn_dist.mean() / (ben_nn_dist.mean() + eps)).item()

    # * largest pairwise distance among Byzantine updates; 0 means every Byzantine client sent an identical vector
    stats['p0_byz_max_pairwise_dist'] = torch.cdist(flat_byz, flat_byz).max().item()

    # * (b) explained-variance spectrum of the benign updates.
    # * the n x n Gram matrix gives the same spectrum as an SVD of the [n x n_params] matrix
    # * without ever touching the millions of coordinates
    ben_mean = flat_ben.mean(dim=0)
    centered_ben = flat_ben - ben_mean
    gram = (centered_ben @ centered_ben.T).double()
    eigvals, eigvecs = torch.linalg.eigh(gram)  # * eigenvalues come back in ascending order
    eigvals = eigvals.flip(0).clamp_min(0)      # * reorder to descending, remove tiny negative noise
    eigvecs = eigvecs.flip(1)
    var_ratio = eigvals / (eigvals.sum() + eps)
    # * centering leaves at most n_ben - 1 non-zero directions, so benign updates live in <= n_ben - 1 dims
    stats['p0_var_ratio_top'] = var_ratio[:min(10, n_ben - 1)].tolist()
    stats['p0_var_cumulative'] = var_ratio.cumsum(dim=0)[:n_ben - 1].tolist()

    # * (c) residual norm outside the top-k benign principal subspace, for each k in ks.
    # * the projection of a centered vector r onto the i-th right singular vector v_i is
    # * (X_c r)·u_i / s_i, so only n-sized vectors are needed and no [n_params x k] basis is built
    k_max = min(max(ks), n_ben - 1)
    sing_vals = eigvals[:k_max].sqrt().clamp_min(eps)
    top_eigvecs = eigvecs[:, :k_max]

    # * benign coordinates in the top-k_max basis (in-sample); rows of the Gram matrix are X_c x_j
    ben_coords = (gram @ top_eigvecs) / sing_vals
    ben_sq_norm = gram.diagonal()
    # * Byzantine coordinates: center with the benign mean, then project the same way
    centered_byz = flat_byz - ben_mean
    byz_coords = ((centered_ben @ centered_byz.T).double().T @ top_eigvecs) / sing_vals
    byz_sq_norm = centered_byz.double().pow(2).sum(dim=1)

    ben_cum_sq = ben_coords.pow(2).cumsum(dim=1)
    byz_cum_sq = byz_coords.pow(2).cumsum(dim=1)
    for k in ks:
        if k > k_max:
            continue  # * not enough benign updates to define a k-dimensional subspace
        ben_res = (ben_sq_norm - ben_cum_sq[:, k - 1]).clamp_min(0).sqrt()
        byz_res = (byz_sq_norm - byz_cum_sq[:, k - 1]).clamp_min(0).sqrt()
        # * NOTE: benign residuals are in-sample (each benign update helped build the subspace),
        # * so they are biased low, and reach 0 at k = n_ben - 1; compare with that in mind
        stats[f'p0_res_ben_mean_k{k}'] = ben_res.mean().item()
        stats[f'p0_res_ben_max_k{k}'] = ben_res.max().item()
        stats[f'p0_res_byz_mean_k{k}'] = byz_res.mean().item()
        # * residual as a fraction of the update's own centered norm (scale-free)
        stats[f'p0_res_ben_frac_k{k}'] = (ben_res / (ben_sq_norm.sqrt() + eps)).mean().item()
        stats[f'p0_res_byz_frac_k{k}'] = (byz_res / (byz_sq_norm.sqrt() + eps)).mean().item()

    return stats


# * ---------------------------------------------------------------------------
# * Pipeline-level selection metrics (new, not part of the starter code).
# * Precision, recall and benign false-positive rate of the updates a defense REJECTED, computed against the
# * simulator's ground truth. Only meaningful for defenses that select or reject updates.
# * ---------------------------------------------------------------------------
def eval_selection(kept_ids, sampled_ids, byz_ids) -> dict:  # * precision / recall / benign FPR of rejected updates
    sampled = set(sampled_ids)  # * every client sampled this round
    kept = set(kept_ids) & sampled  # * clients whose updates the pipeline kept
    rejected = sampled - kept  # * everything else counts as rejected
    byz = sampled & set(byz_ids)  # * sampled Byzantine clients (ground truth)
    ben = sampled - byz  # * sampled benign clients
    tp = len(rejected & byz)  # * Byzantine updates correctly rejected
    fp = len(rejected & ben)  # * benign updates wrongly rejected
    nan = float('nan')  # * undefined metrics are reported as NaN
    return {  # * assemble the metric dict
        'sel_n_rejected': len(rejected),  # * number of rejected updates
        'sel_precision': tp / len(rejected) if len(rejected) > 0 else nan,  # * share of rejections that were Byzantine
        'sel_recall': tp / len(byz) if len(byz) > 0 else nan,  # * share of Byzantine updates that were rejected
        'sel_benign_fpr': fp / len(ben) if len(ben) > 0 else nan,  # * share of benign updates that were rejected
    }
