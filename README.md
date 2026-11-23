<!-- * New file (not from the starter code): project README. -->
# SHIELD: An Adversarial AI Solution For Federated Learning

SHIELD is a defense for federated learning (FL) that filters client updates before aggregation. It is built and tested
against **STRIKE**, a Byzantine attack that exploits gradient skewness. SHIELD never sees the true number of
Byzantine clients: it flags suspicious updates from the updates alone and rejects them.

**Hypothesis.** The evidence for gradient skew in the STRIKE paper comes from 2-D LLE views of client updates. In very
high dimensions the "crowd" of benign updates is less well defined, so forged updates that look in-crowd in a coarse,
low-dimensional view may separate in a richer view. SHIELD looks at every round through two views (a coarse one and a
richer, randomly chosen one) in a nonlinear kernel space and rejects what stands out.

<a href="https://colab.research.google.com/github/edominicduke/SHIELD/blob/main/shield_eval_colab.ipynb">Open the evaluation notebook in Colab</a>

---

## 1. This project builds on the STRIKE starter code

This repository is built on top of the code of the authors of

> Yuchen Liu, Chen Chen, Lingjuan Lyu, Yaochu Jin, Gang Chen.
> *Exploit Gradient Skewness to Circumvent Byzantine Defenses for Federated Learning* (the paper that introduces STRIKE).

The starter code is the master branch of <https://github.com/YuchenLiu-a/byzantine_skew>. It provides the FL simulator,
the datasets, the attacks (including STRIKE, `--attack skew`) and the baseline defenses. SHIELD was developed on top of it
so that it is **compatible with the authors' implementation of STRIKE** and can be tested against it directly.

Conventions used throughout the code:

- **Every file that comes from the starter code carries a header comment saying so.**
- **Every change or addition made is commented, and each such comment starts with `*`.**
- **Defaults keep the starter behavior.** All new features sit behind new flags (`--shield`, `--blind_defenses`,
  `--gpu_fast`, `--profile`, `--phase0_diag`, ...) that default to off, so running without them follows the starter
  code path. (This is by design; have not re-run the paper's experiments to confirm identical numbers.)

### Key files added by this work

| File | What it is |
|---|---|
| [fl/defender/shield.py](fl/defender/shield.py) | The SHIELD defense (kernel PCA, multi-start robust fit, scoring, flagging, kernel selection). |
| [fl/attacker/skew_aware.py](fl/attacker/skew_aware.py) | `skew_aware`: a SHIELD-aware version of STRIKE used to stress-test SHIELD. |
| [shield_eval_colab.ipynb](shield_eval_colab.ipynb) | Colab notebook that runs SHIELD and the baselines under STRIKE and builds the tables and plots. |
| [phase0_colab.ipynb](phase0_colab.ipynb) | Colab notebook for the Phase 0 diagnostics that compared candidate dimensionality approaches before SHIELD was built. |
| `phase0_colab_run.ipynb`, `phase0_results.csv` | Saved output of one Phase 0 run. |
| [requirements.txt](requirements.txt), [README.md](README.md) | Dependencies and this file. |

### Starter files modified (beyond adding the header comment)

| File | Change |
|---|---|
| [main.py](main.py) | New command-line flags for SHIELD, the SHIELD-aware attacker, blind defenses, GPU speed-ups and profiling. |
| [fl/__init__.py](fl/__init__.py) | Blind-defense knowledge, selection metrics (precision, recall, benign FPR), per-phase profiling, `--gpu_fast` switch, SHIELD logging. |
| [fl/defender/defender.py](fl/defender/defender.py) | Runs SHIELD before the aggregator; `commit` flag so attacker queries do not change SHIELD's state; shadow-mode oracle clean path. |
| [fl/attacker/\_\_init\_\_.py](fl/attacker/__init__.py) | Registers `skew_aware`. |
| [fl/attacker/omn.py](fl/attacker/omn.py) | Its defender queries use `commit=False`. |
| [fl/attacker/skew.py](fl/attacker/skew.py) | GPU path for STRIKE's heavy distance computation under `--gpu_fast`. |
| [fl/client/client.py](fl/client/client.py) | GPU-resident local training under `--gpu_fast`. |
| [fl/utils/data.py](fl/utils/data.py) | `make_gpu_loader` (GPU-resident data). |
| [fl/utils/update.py](fl/utils/update.py) | Phase 0 diagnostics, selection metrics, GPU statistics. |
| [fl/utils/\_\_init\_\_.py](fl/utils/__init__.py), [.gitignore](.gitignore) | Exports for the new functions; one ignore-rule change. |

All other starter files (the other attacks and defenses, `configs/`, `fl/utils/nn.py`, ...) only have the header added.

---

## 2. Key design decisions

### 2.1 Kernel PCA with an RBF kernel
Updates are mapped to a higher-dimensional space with an RBF kernel and PCA is performed there. This adds *curved*
structure to the feature space, which makes it more likely to expose malicious updates than a flat (linear) view.
Everything runs on n x n kernel (Gram) matrices, never on matrices as wide as the model. A linear kernel is also
computed and logged alongside the RBF kernels as an ablation; it does not drive rejection.

### 2.2 Robust multi-start fit
SHIELD has to learn what the benign crowd looks like from updates that may include forged ones. It does so with a
**robust multi-start fit** and no density assumption. It runs 16 **starts** per round (plus one extra start from all
updates). A start works like this:

1. Randomly select 4 updates (the start subset).
2. Fit kernel PCA on these 4 updates.
3. Score all updates by their residual from that fit, and keep the half with the smallest residuals. This is the
   start's initial estimate of the benign crowd.
4. **Refit loop** (up to 3 iterations, stopping early once nothing changes): fit kernel PCA on the current crowd, score
   every update (see 2.4), flag outliers, and re-select the best-explained half as the new crowd.
5. The **result of the start** is its final crowd, the scores of all updates, and the flagged outliers.

The starts are independent of each other. **SHIELD keeps the start whose scores separate most cleanly into a benign
group and an outlier group** (an Otsu-style between-group over total variance ratio; ties go to the start with fewer
flags). It uses no labels to decide. Small start subsets are used because a small random subset is much more likely to
be free of forged updates than a large one, so at least one of the 16 usually gives an uncontaminated fit.

### 2.3 Two views: a tracked low view and a random higher view
- **Low view:** the number of kernel-PCA components `k_low` that explains 85% of the variance. It is tracked across
  rounds (median of recent estimates, at most +-2 change per round, and the maximum raw estimate across starts so that
  contaminated starts cannot drag it down).
- **Higher view:** `k_high = factor x k_low`, with `factor` drawn at random each round from {1.5, 2, 3}. The randomness
  means an attacker cannot tune its updates to one specific richer view.
- Caps: the low view keeps at most m/3 components and the higher view at most 3m/4, where m is the size of the fit set.

### 2.4 Three features, one robust threshold
Each update gets three scores:

1. the **whitened low-view distance** (how far it is from the crowd along the main directions, in units of the crowd's spread),
2. the **in-between band**, the energy between `k_low` and `k_high`,
3. the **residual beyond `k_high`**, the part no kept direction explains.

Each feature is turned into a robust z-score (log scale, median and MAD), and an update's score is the largest of the
three. An update is **flagged** when its score exceeds `c = 3.5`. Fit-set members get **5-fold cross-fitted**
(out-of-fold) scores, so an update never helps define the crowd it is judged against.

### 2.5 Semi-random kernel choice
The RBF bandwidth is the median pairwise distance multiplied by a value `v` from a predefined set
`V = {0.25, 0.5, 1, 2, 4}`. The best `v` varies from round to round, so SHIELD chooses it in a semi-random way:

1. Every round, the scores from each `v` are ranked by how well they split the updates into two groups (the same
   separation measure as in 2.2). Ties are broken by temporal stability, i.e. how much the `v`'s flagged set overlaps
   with its own flagged set from the previous round (Jaccard).
2. The top 3 form the pool, and **one is drawn uniformly at random for the next round**. So the kernel used in round
   t+1 was chosen using round t.

The randomness is deliberate: an attacker cannot know which kernel is in play. The cost is that a draw sometimes lands
on a poor kernel.

### 2.6 Hard rejection with a warm-up and a cap
- **Warm-up:** no updates are rejected in the first 3 rounds, to give the crowd structure time to form. Everything is
  still scored and logged.
- **Hard rejection:** after warm-up, flagged updates are removed completely rather than down-weighted. The priority is
  to reduce the performance degradation caused directly by STRIKE, which in life-critical settings (for example
  hospitals) could be harmful.
- **Cap:** at most 50% of the round's updates can be rejected, to stop genuinely benign updates from being ignored in
  excess. If fewer than 8 updates are present, everything is accepted.

### 2.7 Blind to the Byzantine count
SHIELD never receives the true number of Byzantine clients. Ground truth is used only to compute evaluation metrics
(precision, recall, benign false-positive rate), never for a decision. The aggregator that runs after SHIELD is told
there are 0 Byzantine updates.

### 2.8 Shadow mode and active mode
- `--shield shadow`: SHIELD scores and logs, but training follows an oracle clean path (only benign updates are
  aggregated, using simulator ground truth that SHIELD itself never sees). This measures how well SHIELD *can* detect
  STRIKE without the attack or the rejections disturbing training.
- `--shield active`: flagged updates are hard-rejected and the rest are aggregated. This measures what SHIELD actually
  achieves for training.

### 2.9 A separate random generator
The `k_high` factor, the 16 start subsets and the kernel draw all come from SHIELD's own random generator, seeded from
`--seed` (or `--shield_seed`). It never touches the global streams used by client sampling and training, so runs stay
reproducible and comparable when only the defense changes. (In a real deployment this generator should be seeded
from secret entropy.)

### 2.10 A SHIELD-aware attacker
To avoid testing SHIELD against an attacker that does not know it exists, `--attack skew_aware` adapts STRIKE to evade
SHIELD in three independent ways:
- **`--aware_project`** (on by default): keep only the part of STRIKE's deviation that lies inside the span of the benign
  updates (the part SHIELD's residual feature does not see).
- **`--aware_noise X`** (default 0): give each Byzantine client its own in-span noise so the forged copies are not identical.
- **`--aware_oracle`** (on by default): query the defender for accept or reject only (like the Omn attack), and
  bisection-search the largest deviation strength that still gets every forged update accepted
  (`--aware_queries`, default 8). The attacker never reads SHIELD's private generator or its ground-truth bookkeeping.

### 2.11 Fair comparison with blind baselines
SHIELD does not get the true count, so the baselines do not either (`--blind_defenses`). Each baseline receives an
assumed count `floor(ratio x n_sampled)`: **0.5** (equal to SHIELD's rejection cap, so every defense gets the same upper
bound) and **0.3** as a sensitivity check. The true ratio in the experiments is 0.2. SHIELD's cap is only a ceiling (it
rejects what its threshold flags), whereas Krum, Bulyan and DnC remove the assumed count every round, so baselines
at 0.5 are expected to over-reject benign clients.

Only defenses that make a per-update decision can report **precision, recall and benign FPR** of what they rejected:
Krum, Bulyan, Aksel and DnC (and SHIELD in active mode). Median, RFA, CClip, trimmed mean and mean have no such step,
and MDA (which enumerates a very large number of subsets) is infeasible at this client count.

### 2.12 Practical choices
- **Dataset:** CIFAR-10 with a Dirichlet partition (the starter config: AlexNet, 50 clients, all sampled each round,
  20% Byzantine in the evaluation).
- **GPU speed-ups** (`--gpu_fast`): GPU-resident data and client models, GPU statistics and Gram matrices. It falls
  back to the normal path where unsupported (ImageNet12). `--profile` prints a per-phase timing breakdown.
- **Phase 0 diagnostics** (`--phase0_diag`): optional per-round statistics comparing benign and Byzantine updates,
  used before building SHIELD to assess different dimensionality approaches.

---

## 3. Setup and running

### 3.1 Requirements
- Python 3.9 or newer.
- An **NVIDIA GPU with CUDA**: the starter code calls `.cuda()` directly, so training cannot run on a CPU.
- The packages in [requirements.txt](requirements.txt): `torch`, `torchvision`, `numpy`, plus `pandas` and `matplotlib`
  for the notebook's tables and plots, and optionally `wandb` (only imported with `--wandb`).

### 3.2 Run the evaluation notebook on Google Colab (recommended)
The notebook is [shield_eval_colab.ipynb](shield_eval_colab.ipynb). It runs SHIELD and the baselines under STRIKE
and produces the summary table, timing breakdown and plots.

1. **Open it in Colab.** Use the badge at the top of this file, or in Colab choose *File > Open notebook > GitHub*
   and enter `edominicduke/SHIELD`.
2. **Select a GPU:** *Runtime > Change runtime type > T4 GPU*. The first code cell asserts that a GPU is attached.
3. **Get the code (section 2 of the notebook).** The cell clones `REPO_URL`
   (`https://github.com/edominicduke/SHIELD.git`) into `/content/SHIELD`. If you work from a fork, change `REPO_URL`.
   If you leave `REPO_URL` empty, the cell asks you to upload a zip of the project folder instead. The cloned code must
   contain `fl/defender/shield.py`, so commit and push your changes first if you edited the code.
4. **Dependencies and Drive (section 3).** Colab already has `torch`, `torchvision`, `numpy`, `pandas` and `matplotlib`.
   The cell only installs `wandb` (optional). With `USE_DRIVE = True` it mounts Google Drive (you will be asked to
   authorize it) and keeps the run logs in `MyDrive/shield_runs`, so a disconnect does not lose finished runs. Set
   `USE_DRIVE = False` to keep logs on the Colab disk only (they are lost when the runtime is recycled).
5. **Settings (section 4).** The defaults are CIFAR-10, 100 rounds, 20% Byzantine, one seed (`SEED = 0`), Dirichlet
   partition, SHIELD warm-up of 3 rounds, assumed ratios `[0.5, 0.3]`, baselines Krum, Bulyan and Aksel, `GPU_FAST = True`,
   `PROFILE = True` and `PARALLEL = 2`. Edit them as needed. For a quick test set `ROUNDS = 5`, or choose a subset
   with `RUN_NAMES` (for example `['shield_active_strike', 'krum_blind0.5']`).
6. **Run (section 5).** This launches one `main.py` process per experiment and writes each log to
   `<RUN_DIR>/<run name>__<TAG>.log`. The run matrix is:
   - `shield_shadow_strike`, `shield_active_strike`, `shield_active_aware`, `shield_active_aware_noise`
   - `<baseline>_blind<ratio>` for each baseline and assumed ratio (for example `krum_blind0.5`)

   A run is skipped if its log is complete and was made with exactly the same command, so you can re-run the cell
   after a disconnect. Incomplete logs, or logs from other settings, are re-run.
7. **Read the results (sections 6 to 9).** The following cells parse the logs and build the summary table
   (accuracy, precision, recall, benign FPR), the timing table, the accuracy curves, the precision/recall/FPR bars, and
   the per-kernel AUC diagnostics from the shadow run. A summary CSV is written to `/content/shield_eval_summary.csv`
   (download it before the runtime ends).

**Time.** In our T4 logs, a SHIELD run takes roughly 15 to 20 seconds per round, and rounds where the aware attacker's
oracle search runs take about twice as long, so a 100-round SHIELD run takes roughly half an hour to an hour. The full
matrix is a multi-hour job. Colab's free tier can disconnect long sessions, which is why Drive persistence is on by default.

**Reading the metrics.**
- *Accuracy:* lower means the attack did more damage.
- *Recall:* share of the Byzantine updates that were rejected. *Precision:* share of the rejections that were
  Byzantine. *Benign FPR:* share of the honest updates that were rejected.
- Shadow mode trains on benign updates only, so its accuracy is a no-attack reference and is not comparable to the
  other runs. It has no selection metrics by design.
- Results come from one seed unless you add more; differences of a few points are within noise.

### 3.3 Run locally (needs an NVIDIA GPU)
```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.txt     # on a GPU machine, install the CUDA build of PyTorch first (see pytorch.org)

# SHIELD in active mode against STRIKE, CIFAR-10 (the dataset downloads automatically into data/cifar10)
python main.py --dataset cifar10 --framework fedavg --byz_client_ratio 0.2 --partition dirichlet \
    --server_n_epochs 100 --seed 0 --devices 0 --nice --gpu_fast \
    --aggregator mean --attack skew --shield active

# the same against the SHIELD-aware attacker (add --aware_noise 0.5 for the noisy variant)
python main.py --dataset cifar10 --framework fedavg --byz_client_ratio 0.2 --partition dirichlet \
    --server_n_epochs 100 --seed 0 --devices 0 --nice --gpu_fast \
    --aggregator mean --attack skew_aware --shield active

# a blind baseline (assumed Byzantine ratio 0.5)
python main.py --dataset cifar10 --framework fedavg --byz_client_ratio 0.2 --partition dirichlet \
    --server_n_epochs 100 --seed 0 --devices 0 --nice --gpu_fast \
    --aggregator krum --attack skew --blind_defenses --assumed_byz_ratio 0.5
```
`--nice` skips the starter code's GPU-memory check (`utils.block`), which is not needed on Colab or a free GPU.
`--aggregator mean` means SHIELD is the only defense. Each round prints its metrics, and with `--shield` it also prints
a SHIELD section (selected multiplier, `k_low`, `k_high`, winning start, precision, recall, benign FPR).

Main SHIELD flags (all in [main.py](main.py)): `--shield {off,shadow,active}`, `--shield_c` (3.5),
`--shield_warmup` (3), `--shield_max_reject_frac` (0.5), `--shield_multipliers` (`0.25,0.5,1,2,4`),
`--shield_pool_size` (3), `--shield_khigh_factors` (`1.5,2,3`), `--shield_n_starts` (16), `--shield_seed_size` (4),
`--shield_n_folds` (5), `--shield_seed`. Note that `--shield_seed_size` is the number of updates in each start subset,
not a random-number seed.

---

## 4. Known limitations
- **The aware attacker's oracle does not replay the real round's random draws.** Its queries to SHIELD use the previous
  `k_high` factor and start subsets drawn from a copy of the generator, which differ from the draws of the real round.
  The aware attacker is therefore weaker than designed, and SHIELD looks stronger against it than it should.
- **Benign false positives and kernel randomness:** SHIELD sometimes rejects only benign clients in a round, and the
  random draw from the pool sometimes picks a poorly separating kernel.
- **Recall is low** compared with some blind baselines (which over-reject); precision and benign FPR are better.
- **Single seed, single dataset, single partition** in the reported runs.
- **Training speed** is bounded by the starter's model and training settings (AlexNet, lr 0.1, momentum 0.9), not by SHIELD.
