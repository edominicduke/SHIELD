"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
from argparse import Namespace
cifar100_config = Namespace(
    data_root = 'data/cifar100',
    n_clients = 50,
    n_classes = 100,

    architecture = 'squeezenet',
    loss = 'cross_entropy',
    server_n_epochs = 300,
    n_sampled_clients = 50,

    client_batch_size = 64,
    client_optimizer = 'sgd',
    client_lr = 0.1,
    client_momentum = 0.5,
    client_weight_decay = 1e-4,
    client_n_epochs = 1,
    clip_max_norm = 2.0,
)

if __name__ == '__main__':
    print(vars(cifar100_config))
