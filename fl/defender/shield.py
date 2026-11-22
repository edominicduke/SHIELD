# * New file (not from the starter code): SHIELD, a blind kernel-subspace filter for client updates.
# * Pipeline each round (all of it runs on n x n kernel matrices, never on n x n_params basis matrices):
# *   1. build an RBF kernel at several bandwidth multipliers (plus a linear kernel, logged as an ablation)
# *   2. per kernel: robust iterative refit of a kernel-PCA subspace on the unflagged updates, run from several
# *      random small starting subsets (multi-start, no density assumption); the start whose fit separates best wins
# *   3. score every update with (a) whitened distance in a low view, (b) whitened energy in the band between
# *      the low view and a randomly chosen higher view, (c) residual outside the higher view
# *   4. flag by a robust (median + c * MAD) rule on the log scores, capped at a fraction of the updates
# *   5. rank the multipliers by separation (tie-break: temporal stability); the top ones form next round's pool,
# *      and one pool member is drawn at random (separate generator) to drive the next round's flagging
from __future__ import annotations

import copy
import math
import numpy as np
import torch

from argparse import Namespace
from ..utils import flatten_updates

# * small constant to avoid division by zero and log(0)
_EPS = 1e-8
# * view-size caps relative to the fit-set size m: the low view keeps at most m / 3 components and the higher view
# * at most 3m / 4, so the band and the residual always have room (near the full rank they collapse to numerical noise)
_LOW_CAP_FRAC = 1.0 / 3.0
_HIGH_CAP_FRAC = 0.75


def _kpca_fit(K: np.ndarray, fit_idx: np.ndarray):
    # * kernel-center the kernel matrix restricted to the fit set and eigendecompose it
    # * restrict the full n x n kernel matrix to the rows/columns of the fit set
    K_ff = K[np.ix_(fit_idx, fit_idx)]
    # * column means (= row means, K is symmetric) and the grand mean, used for feature-space centering
    col_mean = K_ff.mean(axis=0)
    grand = K_ff.mean()
    # * centered kernel matrix: Kc = H K H with H = I - 11^T / m
    Kc = K_ff - col_mean[None, :] - col_mean[:, None] + grand
    # * eigendecomposition of the symmetric centered matrix
    mu, A = np.linalg.eigh(Kc)
    # * eigh returns ascending order; reorder to descending and clip tiny negative numerical noise
    mu = np.clip(mu[::-1], 0.0, None)
    A = A[:, ::-1]
    return mu, A, col_mean, grand


def _kpca_scores(K, fit_idx, mu, A, col_mean, grand, k_low, k_high, var_floor_frac=1e-3):
    # * score ALL n points (fit-set members and others) against the subspace fitted on fit_idx
    n = K.shape[0]  # * number of points being scored
    m = len(fit_idx)  # * number of points the subspace was fitted on
    # * only keep components with non-negligible eigenvalue (a centered m-point kernel has rank <= m - 1)
    n_valid = int((mu > 1e-10 * max(mu[0], _EPS)).sum()) if len(mu) > 0 else 0
    # * the largest number of components we can actually use
    kmax = int(min(k_high, m - 1, n_valid))
    if kmax < 1:
        # * degenerate fit (e.g. all points identical): return all-zero scores so nothing gets flagged
        zeros = np.zeros(n)
        return {'maha_low': zeros, 'maha_high': zeros, 'band': zeros, 'resid': zeros}
    # * the low view cannot exceed the usable number of components
    k_low_eff = int(min(k_low, kmax))

    # * center the cross-kernel rows k(x, fit set) the same way as the fit set
    k_x = K[:, fit_idx]
    row_mean = k_x.mean(axis=1)
    kc_x = k_x - row_mean[:, None] - col_mean[None, :] + grand
    # * squared feature-space length of each centered point
    kc_xx = np.diag(K) - 2.0 * row_mean + grand

    # * projection of each point onto the top-kmax unit-norm feature-space axes
    p = kc_x @ A[:, :kmax] / np.sqrt(mu[:kmax])[None, :]
    # * variance of each axis over the fit set is mu_i / m; floor it so weak axes do not blow up the whitening
    var = mu[:kmax] / m
    var = np.maximum(var, var_floor_frac * var[0])
    # * cumulative sum of squared whitened coordinates (squared Mahalanobis distance after k components)
    cum_z2 = np.cumsum(p ** 2 / var[None, :], axis=1)

    # * whitened distance in the low view and in the higher view
    maha_low = np.sqrt(cum_z2[:, k_low_eff - 1])
    maha_high = np.sqrt(cum_z2[:, kmax - 1])
    # * whitened energy in the band between the two views (components k_low+1 .. k_high)
    band = np.sqrt(np.maximum(cum_z2[:, kmax - 1] - cum_z2[:, k_low_eff - 1], 0.0))
    # * residual outside the higher view, as a fraction of the centered feature-space length
    resid_sq = np.maximum(kc_xx - np.cumsum(p ** 2, axis=1)[:, kmax - 1], 0.0)
    resid = np.sqrt(resid_sq / (np.maximum(kc_xx, 0.0) + _EPS))
    return {'maha_low': maha_low, 'maha_high': maha_high, 'band': band, 'resid': resid}


def _crossfit_feats(K, fit_idx, k_low, k_high, n_folds, feats):
    # * cross-fitting: replace the in-sample scores of fit-set members by out-of-fold scores, so every update is
    # * scored against a subspace it did not help build (removes the in-sample bias that makes fit members look
    # * artificially typical). Folds are assigned by position, so no random numbers are used
    pos = np.arange(len(fit_idx)) % n_folds  # * fold id of each fit-set member
    for f in range(n_folds):
        held = fit_idx[pos == f]  # * members scored out-of-fold in this round of the loop
        train = fit_idx[pos != f]  # * the rest build the subspace
        m_tr = len(train)
        if len(held) == 0 or m_tr < 4:
            continue  # * too few points to cross-fit; keep the in-sample scores for these members
        # * view sizes for the smaller training set, never above the full-fit sizes
        kl = int(min(k_low, max(1, int(m_tr * _LOW_CAP_FRAC))))
        kh = int(min(k_high, max(kl + 1, min(m_tr - 2, int(m_tr * _HIGH_CAP_FRAC)))))
        mu, A, col_mean, grand = _kpca_fit(K, train)  # * subspace without the held-out fold
        sub = _kpca_scores(K, train, mu, A, col_mean, grand, kl, max(kh, kl + 1))
        for name in feats:
            feats[name][held] = sub[name][held]  # * overwrite only the held-out members
    return feats


def _robust_z(values: np.ndarray, ref_mask: np.ndarray, mad_floor: float = 0.05) -> np.ndarray:
    # * log-transform (scores are positive and right-skewed), then z-score with median and scaled MAD of the reference set
    t = np.log(values + _EPS)  # * log scale tames the right skew
    ref = t[ref_mask]  # * reference set = the updates currently trusted (the fit set)
    med = np.median(ref)  # * robust center
    mad = 1.4826 * np.median(np.abs(ref - med))  # * scaled MAD, comparable to a standard deviation
    # * floor the scale so a degenerate (near-constant) reference set cannot produce huge z-scores
    mad = max(mad, mad_floor)
    return (t - med) / mad


def _separation(s: np.ndarray, c: float, max_frac: float) -> float:
    # * Otsu-style separation: best between-class / total variance ratio over splits of the sorted scores
    # * the high group may hold at most max_frac of the updates, and must have mean z >= c (absolute-gap guard)
    n = len(s)  # * number of scores
    srt = np.sort(s)  # * ascending, so every split is "low part | high part"
    total_var = srt.var()  # * total variance of the scores
    if total_var < _EPS:
        # * all scores equal: nothing separates from anything
        return 0.0
    best = 0.0  # * best separation found so far
    max_high = max(1, int(math.floor(max_frac * n)))  # * the high group may not exceed the rejection cap
    for j in range(n - max_high, n):  # * j = index where the high group starts
        if j < 2:
            # * need at least two points in the low group
            continue
        low, high = srt[:j], srt[j:]  # * the two groups for this split
        # * guard: a split only counts if the high group stands out in absolute terms
        if high.mean() < c:
            continue
        w_l, w_h = len(low) / n, len(high) / n  # * group weights
        # * between-class variance divided by total variance (Otsu's criterion, in [0, 1])
        eta = w_l * w_h * (low.mean() - high.mean()) ** 2 / total_var
        best = max(best, float(eta))
    return best


def _jaccard(a: set, b: set) -> float:
    # * overlap of two flagged-id sets; two empty sets count as perfectly stable
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)


def _auc(scores: np.ndarray, is_byz: np.ndarray) -> float:
    # * ground-truth AUC (Byzantine vs benign) with ties counted as half; EVALUATION ONLY
    pos, neg = scores[is_byz], scores[~is_byz]  # * scores of the Byzantine and the benign updates
    if len(pos) == 0 or len(neg) == 0:
        # * AUC is undefined when one group is empty
        return float('nan')
    diff = pos[:, None] - neg[None, :]  # * every Byzantine-vs-benign score difference
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


class Shield:
    def __init__(self, args: Namespace, byz_client_idxs: set[str]) -> None:
        self.args = args  # * keep the full argument namespace
        self.mode = args.shield  # * 'shadow' or 'active'
        self.c = float(args.shield_c)  # * robust z-score flagging cutoff
        self.warmup = int(args.shield_warmup)  # * rounds with no rejection
        self.cap_frac = float(args.shield_max_reject_frac)  # * max fraction of updates rejected per round
        self.var_thresh = float(args.shield_var_thresh)  # * cumulative variance share that defines k_low
        self.refit_iters = int(args.shield_refit_iters)  # * max refit iterations per kernel per round
        self.min_clients = 8  # * below this many updates the subspace is meaningless, so accept everything
        self.fast = bool(getattr(args, 'gpu_fast', False)) and torch.cuda.is_available()  # * GPU Gram matrix switch
        # * multi-start: number of random starting subsets, and the size of each (small subsets are likely to be
        # * free of attackers, which is what lets at least one start give an uncontaminated fit)
        self.n_starts = int(getattr(args, 'shield_n_starts', 16))  # * default 16 starts
        self.seed_size = int(getattr(args, 'shield_seed_size', 4))  # * default 4-point start subsets
        self.n_folds = int(getattr(args, 'shield_n_folds', 5))  # * folds for cross-fitting the scores
        # * bandwidth multipliers (times the median pairwise distance) and the k_high factors (times k_low)
        self.multipliers = [float(x) for x in str(args.shield_multipliers).split(',')]
        self.factors = [float(x) for x in str(args.shield_khigh_factors).split(',')]
        # * names of the RBF kernels, e.g. 'rbf0.25', used as dictionary keys and in logs
        self.keys = [f'rbf{m:g}' for m in self.multipliers]
        # * number of top multipliers kept as next round's candidate pool
        self.pool_size = max(1, min(int(args.shield_pool_size), len(self.keys)))

        # * SEPARATE random generator for the k_high factor and the multiplier draw. It is a private numpy
        # * Generator, so it never touches the global numpy/torch/random streams used by client sampling and
        # * training. In the simulator it is seeded deterministically for reproducibility; a real deployment
        # * should seed it from secret entropy so an attacker cannot predict the draws.
        seed = args.shield_seed if args.shield_seed is not None else int(args.seed) * 1000003 + 12345
        self._rng = np.random.default_rng(seed)

        # * ground-truth Byzantine ids: used ONLY to compute evaluation metrics, never for any decision
        self._eval_byz_idxs = set(byz_client_idxs)

        # * state carried across rounds
        self.t = 0                                    # * number of committed rounds
        self._pool = list(self.keys)                  # * round 0: every multiplier is a candidate
        self._selected = self.keys[int(self._rng.integers(len(self.keys)))]  # * multiplier driving the current round
        self._last_factor = self.factors[len(self.factors) // 2]  # * fallback factor for non-committing queries
        self._klow_hist: dict[str, list[int]] = {}    # * per-kernel history of raw k_low estimates
        self._klow_last: dict[str, int] = {}          # * per-kernel last used (smoothed) k_low
        self._prev_flagged: dict[str, set] = {}       # * per-kernel flagged ids from the previous round

    def _smooth_k_low(self, key: str, raw: int, m: int) -> int:
        # * median over the last few raw estimates, rate-limited to +-2 per round, clipped to what the fit set allows
        hist = self._klow_hist.get(key, [])  # * past raw estimates for this kernel
        window = hist[-4:] + [raw]  # * last four plus the current one
        k = int(round(float(np.median(window))))  # * median is robust to a single noisy estimate
        if key in self._klow_last:
            # * rate limit: k_low may move at most 2 per round (limits slow poisoning of the estimate)
            prev = self._klow_last[key]
            k = int(np.clip(k, prev - 2, prev + 2))
        return int(np.clip(k, 1, max(1, int(m * _LOW_CAP_FRAC))))

    def _run_kernel(self, K: np.ndarray, key: str, factor: float, init_mask: np.ndarray) -> dict:
        # * robust iterative refit for one kernel: fit -> score -> flag -> refit on unflagged -> rescore all
        n = K.shape[0]  # * number of updates this round
        # * never reject more than cap_frac of the updates, and always keep at least 3 for the fit
        max_rej = max(0, min(int(math.floor(self.cap_frac * n)), n - 3))
        fit_mask = init_mask.copy()  # * the first fit uses the starting subset handed in by the multi-start wrapper
        prev_flags = None  # * flags from the previous iteration, for the convergence check
        out = {}  # * results of the latest iteration
        n_iters = 0  # * iterations actually run
        for _ in range(self.refit_iters):
            n_iters += 1
            fit_idx = np.where(fit_mask)[0]  # * indices of the updates used for this fit
            m = len(fit_idx)  # * fit-set size
            mu, A, col_mean, grand = _kpca_fit(K, fit_idx)  # * kernel PCA on the fit set
            # * raw k_low: smallest k whose cumulative variance share reaches var_thresh on the current fit set
            cum = np.cumsum(mu / (mu.sum() + _EPS))
            raw_k_low = int(np.clip(np.searchsorted(cum, self.var_thresh) + 1, 1, max(1, int(m * _LOW_CAP_FRAC))))
            k_low = self._smooth_k_low(key, raw_k_low, m)  # * tracked, smoothed low-view size
            # * higher view: randomly drawn factor times k_low, kept strictly above k_low and below the fit rank
            k_high_cap = max(k_low + 1, min(m - 2, int(m * _HIGH_CAP_FRAC)))
            k_high = int(np.clip(round(factor * k_low), k_low + 1, k_high_cap))

            # * score all n updates (fit-set members and the rest) against this fit
            feats = _kpca_scores(K, fit_idx, mu, A, col_mean, grand, k_low, k_high)
            # * cross-fit: fit-set members get out-of-fold scores so nobody is scored in-sample
            feats = _crossfit_feats(K, fit_idx, k_low, k_high, self.n_folds, feats)
            # * robust z-scores of the three differentiating features. The reference is ALL updates (median and MAD
            # * are robust below 50% contamination, which the cap enforces); referencing the small, tightly
            # * clustered fit set instead would make every outside point look like an outlier
            all_mask = np.ones(n, dtype=bool)
            z_low = _robust_z(feats['maha_low'], all_mask)
            z_band = _robust_z(feats['band'], all_mask)
            z_res = _robust_z(feats['resid'], all_mask)
            # * flag score = the worst (largest) of the three z-scores
            s = np.maximum.reduce([z_low, z_band, z_res])

            flags = s > self.c  # * flag everything above the cutoff
            # * enforce the rejection cap: keep only the highest-scoring updates
            if flags.sum() > max_rej:
                keep = np.argsort(-s)[:max_rej]
                flags = np.zeros(n, dtype=bool)
                flags[keep] = True
            # * remember this iteration's results
            out = {
                's': s, 'z_low': z_low, 'z_band': z_band, 'z_res': z_res, 'flags': flags,
                'k_low': k_low, 'k_high': k_high, 'raw_k_low': raw_k_low, 'fit_size': m,
            }
            # * concentration step (as in robust-statistics MCD): the next fit set is the h = ceil(n / 2) updates
            # * best explained by the current fit (lowest score s). Refitting on "everything unflagged" would pull
            # * undetected attackers back into the fit and mask them; keeping a fixed half does not, and it selects
            # * by fit quality, not by density
            h = int(math.ceil(0.5 * n))
            next_mask = np.zeros(n, dtype=bool)
            next_mask[np.argsort(s)[:h]] = True
            # * stop early once neither the flagged set nor the fit set changes
            if prev_flags is not None and np.array_equal(flags, prev_flags) and np.array_equal(next_mask, fit_mask):
                break
            prev_flags = flags
            fit_mask = next_mask
        out['iters'] = n_iters
        return out

    def _seed_mask(self, K: np.ndarray, seed_idx: np.ndarray):
        # * turn a small random seed subset into a starting fit set: fit the seed, then keep the h = ceil(n / 2)
        # * updates that the seed's subspace explains best (smallest residual). Selection is by fit to the seed's
        # * subspace, not by how dense a region is
        n = K.shape[0]  # * number of updates
        h = int(math.ceil(0.5 * n))  # * size of the expanded starting set
        mu, A, col_mean, grand = _kpca_fit(K, seed_idx)  # * kernel PCA on the seed only
        r = len(seed_idx) - 1  # * a centered m0-point kernel has rank at most m0 - 1
        # * score everything against the seed subspace using all of its components
        feats = _kpca_scores(K, seed_idx, mu, A, col_mean, grand, max(1, r // 2), r)
        if not np.any(feats['resid'] > 0):
            # * degenerate seed (e.g. identical points): this start is unusable
            return None
        mask = np.zeros(n, dtype=bool)
        mask[np.argsort(feats['resid'])[:h]] = True  # * keep the h best-explained updates
        return mask

    def _run_kernel_multistart(self, K: np.ndarray, key: str, factor: float, seeds: list) -> dict:
        # * run the robust refit from the full set (start -1) and from every random seed, keep the best start
        n = K.shape[0]
        runs = [(-1, self._run_kernel(K, key, factor, np.ones(n, dtype=bool)))]  # * the plain full-data start
        for si, seed_idx in enumerate(seeds):
            init = self._seed_mask(K, seed_idx)  # * expanded starting set for this seed
            if init is None:
                continue  # * skip degenerate seeds
            runs.append((si, self._run_kernel(K, key, factor, init)))

        # * choose the start by an UNLABELED criterion: how cleanly its final scores separate (bucketed to 0.05),
        # * then fewer flags (when nothing separates, prefer the fit that flags least), then the first one
        def crit(run):
            out = run[1]
            return (-round(_separation(out['s'], self.c, self.cap_frac) / 0.05), int(out['flags'].sum()))
        best_start, best = min(runs, key=crit)
        best['start'] = best_start  # * which start won (-1 = full-data start)
        best['n_starts_run'] = len(runs)  # * how many usable starts were tried
        # * k_low estimate that feeds the cross-round history: the MAXIMUM raw estimate over all starts, not the
        # * winner's. A contaminated fit is dominated by the attackers' cluster, which makes its spectrum look
        # * low-dimensional and can only LOWER the raw estimate, and most random starts are contaminated (a 6-point
        # * seed is clean only about a quarter of the time), so even a high percentile lands on contaminated values.
        # * The max keeps contaminated starts from dragging the tracked k_low down, which would weaken later
        # * detection (verified: with k_low = 3 clean starts miss the forged updates, with k_low >= 6 they catch all).
        # * An attacker could try to push the estimate UP instead; the +-2 per round rate limit bounds that
        best['raw_k_low_round'] = int(max(r[1]['raw_k_low'] for r in runs))
        return best

    def process(self, client_msgs: dict, commit: bool = True):
        # * returns (accepted client ids, log dict). commit=False is for attacker queries (e.g. the omniscient
        # * attack): it must not draw random numbers or change any state
        updates = {idx: msg['update'] for idx, msg in client_msgs.items()}  # * per-client update dicts
        flat, structure = flatten_updates(updates)  # * [n_clients x n_params] matrix
        ids: list[str] = structure['client_idxs']  # * client ids in row order
        n = len(ids)
        if n < self.min_clients:
            # * too few updates for a meaningful subspace: accept everything
            return ids, {'shield_skipped': 1}

        # * center the updates (distances are unchanged) and build the linear Gram matrix with one matmul
        if self.fast:
            # * GPU path: one transfer, then center and form the Gram matrix on the GPU
            flat_g = flat.cuda()
            centered = flat_g - flat_g.mean(dim=0, keepdim=True)
            G = (centered @ centered.T).double().cpu().numpy()
        else:
            centered = flat - flat.mean(dim=0, keepdim=True)
            G = (centered @ centered.T).double().numpy()
        # * squared distances from the Gram matrix; clamp tiny negatives from float error
        diag = np.diag(G)
        D2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * G, 0.0)
        np.fill_diagonal(D2, 0.0)  # * exact zeros on the diagonal
        D = np.sqrt(D2)  # * pairwise distances
        # * median heuristic: median of the off-diagonal pairwise distances
        med = float(np.median(D[~np.eye(n, dtype=bool)]))
        med = max(med, _EPS)

        # * kernels: the linear one is logged as an ablation, the RBFs at each multiplier form the pool
        kernels = {'lin': G}
        for mult, key in zip(self.multipliers, self.keys):
            sigma = mult * med  # * bandwidth = multiplier times the median distance
            kernels[key] = np.exp(-D2 / (2.0 * sigma ** 2))  # * RBF kernel matrix

        # * draw the k_high factor for this round (or reuse the last one for non-committing queries)
        if commit:
            factor = float(self.factors[int(self._rng.integers(len(self.factors)))])
            self._last_factor = factor
        else:
            factor = self._last_factor

        # * draw the random starting subsets with the separate generator; a non-committing query uses a COPY of the
        # * generator so it sees the same kind of draw without advancing the real stream
        rng = self._rng if commit else copy.deepcopy(self._rng)
        m0 = max(3, min(self.seed_size, n - 2))  # * seed size, kept valid for small n
        seeds = [rng.choice(n, size=m0, replace=False) for _ in range(self.n_starts)]  # * shared by all kernels

        # * run the multi-start robust refit and scoring for every kernel
        results = {name: self._run_kernel_multistart(K, name, factor, seeds) for name, K in kernels.items()}

        # * the multiplier selected last round drives this round's flagging
        sel = self._selected
        sel_flags = results[sel]['flags']  # * boolean flags from the selected kernel
        in_warmup = self.t < self.warmup  # * warm-up rounds reject nothing
        # * during warm-up nothing is rejected, but everything is still scored and logged
        rejected = np.zeros(n, dtype=bool) if in_warmup else sel_flags
        accepted_ids = [i for i, r in zip(ids, rejected) if not r]  # * ids that survive hard rejection

        # * ---- evaluation metrics (ground truth, logging only) ----
        is_byz = np.array([i in self._eval_byz_idxs for i in ids])  # * ground truth, never used for decisions
        n_flag = int(sel_flags.sum())  # * number flagged
        tp = int((sel_flags & is_byz).sum())  # * Byzantine updates correctly flagged
        fp = int((sel_flags & ~is_byz).sum())  # * benign updates wrongly flagged
        log = {
            'shield_selected': sel,  # * multiplier used this round
            'shield_pool': list(self._pool),  # * candidate pool this round was drawn from
            'shield_warmup': int(in_warmup),  # * 1 while in warm-up
            'shield_factor': factor,  # * k_high factor drawn this round
            'shield_median_dist': med,  # * median pairwise distance (the bandwidth scale)
            'shield_n_flagged': n_flag,  # * number flagged by the selected kernel
            'shield_n_rejected': int(rejected.sum()),  # * number actually rejected
            'shield_k_low': results[sel]['k_low'],  # * low-view size of the selected kernel
            'shield_k_high': results[sel]['k_high'],  # * higher-view size of the selected kernel
            'shield_fit_size': results[sel]['fit_size'],  # * size of the final fit set
            'shield_start': results[sel]['start'],  # * winning start for the selected kernel (-1 = full-data start)
            'shield_n_starts': results[sel]['n_starts_run'],  # * usable starts tried
            'shield_precision': tp / n_flag if n_flag > 0 else float('nan'),
            'shield_recall': tp / int(is_byz.sum()) if is_byz.sum() > 0 else float('nan'),
            'shield_benign_fpr': fp / int((~is_byz).sum()) if (~is_byz).sum() > 0 else float('nan'),
            'shield_flagged_ids': [i for i, f in zip(ids, sel_flags) if f],
        }
        # * per-kernel, per-feature ground-truth AUC, so the logs show which kernel/feature separates STRIKE
        for name, res in results.items():
            for feat in ('s', 'z_low', 'z_band', 'z_res'):
                log[f'shield_auc_{name}_{feat}'] = _auc(res[feat], is_byz)
            log[f'shield_klow_{name}'] = res['k_low']  # * low-view size per kernel
            log[f'shield_start_{name}'] = res['start']  # * which start won per kernel (-1 = full-data start)
            log[f'shield_nflag_{name}'] = int(res['flags'].sum())  # * how many each kernel flagged

        # * ---- rank the multipliers by separation, tie-break temporal stability ----
        # * flagged-id sets of every kernel this round, for the stability measure
        cur_flagged = {name: {i for i, f in zip(ids, res['flags']) if f} for name, res in results.items()}
        seps, stabs = {}, {}  # * separation and stability per RBF multiplier
        for key in self.keys:
            seps[key] = _separation(results[key]['s'], self.c, self.cap_frac)
            prev = self._prev_flagged.get(key)  # * this kernel's flagged set last round
            stabs[key] = _jaccard(prev, cur_flagged[key]) if prev is not None else 0.0
            log[f'shield_sep_{key}'] = seps[key]
            log[f'shield_stab_{key}'] = stabs[key]

        if commit:
            # * rank: separation (bucketed to 0.05 so near-ties fall through) first, then stability
            # * shuffle first (separate generator) so exact ties are broken randomly instead of by list order;
            # * python's sort is stable, so the shuffled order only matters among exact ties
            shuffled = list(self.keys)
            self._rng.shuffle(shuffled)
            ranked = sorted(shuffled, key=lambda k: (-round(seps[k] / 0.05), -stabs[k]))
            self._pool = ranked[:self.pool_size]  # * top-m multipliers become next round's pool
            # * next round's multiplier is drawn uniformly from the pool with the separate generator
            self._selected = self._pool[int(self._rng.integers(len(self._pool)))]
            # * update per-kernel state (k_low tracker and flagged-id history)
            for name, res in results.items():
                self._klow_hist.setdefault(name, []).append(res['raw_k_low_round'])  # * robust per-round estimate
                self._klow_last[name] = res['k_low']
                self._prev_flagged[name] = cur_flagged[name]
            self.t += 1  # * one more committed round
            log['shield_next_pool'] = list(self._pool)
            log['shield_next_selected'] = self._selected

        return accepted_ids, log
