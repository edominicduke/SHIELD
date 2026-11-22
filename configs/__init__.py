"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
from .femnist import femnist_config
from .cifar10 import cifar10_config
from .cifar100 import cifar100_config
from .imagenet12 import imagenet12_config

__all__ = [
    'femnist_config', 
    'cifar10_config', 
    'cifar100_config', 
    'imagenet12_config', 
]