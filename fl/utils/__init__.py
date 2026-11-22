"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
from .data import load_data, ClientDataset, make_gpu_loader  # * make_gpu_loader: GPU-resident data (--gpu_fast)
from .nn import get_model, get_loss
from .optim import get_optimizer, step
from .update import flatten_update, flatten_updates, unflatten_update, eval_ben_updates, eval_agg_dev, eval_byz_dev, eval_phase0_diagnostics, eval_selection, set_fast  # * Phase 0 diagnostics, selection metrics, GPU switch
from .utils import block, eval_perf, bisection, line_maximize, print_dict, setup_seed

__all__ = [
    'load_data', 'ClientDataset', 'make_gpu_loader',  # * make_gpu_loader is new
    'get_model', 'get_loss',
    'get_optimizer', 'step', 
    'flatten_update', 'flatten_updates', 'unflatten_update', 'eval_ben_updates' , 'eval_agg_dev', 'eval_byz_dev', 'eval_phase0_diagnostics', 'eval_selection', 'set_fast',  # * Phase 0 diagnostics, selection metrics, GPU switch
    'block', 'eval_perf', 'bisection', 'line_maximize', 'print_dict', 'setup_seed',
]