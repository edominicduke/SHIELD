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

from ..client import Client
from .bitflip import BitFlip
from .ipm import Ipm
from .lie import Lie
from .mimic import Mimic
from .minmax import MinMax
from .minsum import MinSum
from .omn import Omn
from .skew import Skew
from .skew_aware import SkewAware  # * SHIELD-aware STRIKE

def get_attacker(byz_clients: dict[str, Client], args: Namespace):
    if args.attack == 'bitflip':
        return BitFlip(byz_clients, args)
    elif args.attack == 'ipm':
        return Ipm(byz_clients, args)
    elif args.attack == 'lie':
        return Lie(byz_clients, args)
    elif args.attack == 'mimic':
        return Mimic(byz_clients, args)
    elif args.attack == 'minmax':
        return MinMax(byz_clients, args)
    elif args.attack == 'minsum':
        return MinSum(byz_clients, args)
    elif args.attack == 'omn':
        return Omn(byz_clients, args)
    elif args.attack == 'skew':
        return Skew(byz_clients, args)
    elif args.attack == 'skew_aware':  # * SHIELD-aware STRIKE
        return SkewAware(byz_clients, args)
    else:
        raise Exception('invalid attack')

__all__ = [
    'get_attacker'
]