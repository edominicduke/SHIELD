"""
The following file is from the master branch of https://github.com/YuchenLiu-a/byzantine_skew,
the repository used for the "Exploit Gradient Skewness to Circumvent Byzantine Defenses for
Federated Learning" paper, which introduces STRIKE. The intention is to use this as starter code
to build off of, so that my implementation of SHIELD (my solution for making STRIKE compatible
and testable against this code) can be developed on top of it.
Any additional changes aside from this will be commented starting with a *
"""
import torch

from argparse import Namespace
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import time

from ..utils import get_loss, get_model, get_optimizer, make_gpu_loader  # * make_gpu_loader: GPU-resident data (--gpu_fast)

# * device used by the --gpu_fast path (module-level so it can be pointed at 'cpu' in tests)
_FAST_DEVICE = 'cuda'  # * GPU device for the fast path

class Client:
    def __init__(self, data: Dataset, args: Namespace):
        self.data = data
        self.args = args
        self.local_model = get_model(self.args.architecture, self.args.n_classes)
        self.loss = get_loss(self.args.loss)
        self.dataloader = DataLoader(self.data, batch_size=self.args.client_batch_size, shuffle=True, num_workers=self.args.n_workers, pin_memory=self.args.pin_memory)
        # * --gpu_fast: GPU-resident data and a model that stays on the GPU between rounds (no per-round CUDA transfers)
        self.fast = False  # * fast path off unless the dataset supports it
        if getattr(self.args, 'gpu_fast', False):  # * only when --gpu_fast is set
            gpu_loader = make_gpu_loader(self.data, self.args.client_batch_size, shuffle=True, device=_FAST_DEVICE)  # * None for unsupported datasets
            if gpu_loader is not None:  # * None means the dataset is unsupported (e.g. ImageNet12): keep the normal path
                self.dataloader = gpu_loader  # * use the GPU-resident loader
                self.local_model.to(_FAST_DEVICE)  # * the model lives on the GPU for the whole run
                self.fast = True  # * fast path on

    def _local_update_fast(self, server_message: dict):  # * GPU-fast local update
        # * --gpu_fast version of local_update: same training steps, but the data and the model are already on the
        # * GPU, and only the finished update (a small dict of tensors) is copied back to the CPU, once
        global_state = server_message['model_state']  # * global weights sent by the server (CPU tensors)
        self.local_model.load_state_dict(global_state)  # * copies the CPU global weights into the GPU-resident model
        optimizer = get_optimizer(self.args.client_optimizer, self.local_model.parameters(), self.args.client_lr, self.args.client_momentum, self.args.client_weight_decay)  # * fresh optimizer per round, as in the starter code
        self.local_model.train()  # * training mode
        for _ in range(self.args.client_n_epochs):  # * local epochs
            for (inputs, targets) in self.dataloader:  # * batches are already on the GPU
                # * drop batch of size 1 when there is batch normalization (same rule as the starter code)
                if self.args.architecture in ['resnet18', 'squeezenet'] and len(inputs) == 1:  # * batch-norm models cannot train on a batch of one
                    continue  # * skip that batch
                optimizer.zero_grad()  # * clear gradients
                outputs = self.local_model(inputs)  # * forward pass on the GPU
                loss = self.loss(outputs, targets)  # * loss on the GPU
                loss.backward()  # * backward pass
                if self.args.clip_max_norm is not None:  # * optional gradient clipping, as in the starter code
                    clip_grad_norm_(self.local_model.parameters(), max_norm=self.args.clip_max_norm)  # * clip the gradient norm
                optimizer.step()  # * SGD step
        with torch.no_grad():  # * no gradients while computing the update
            self.state_update = {}  # * update dict, same format as the starter code
            local_state = self.local_model.state_dict()  # * trained weights, still on the GPU
            for key in local_state:  # * every state tensor
                # * delta computed on the GPU, then moved to the CPU once per tensor
                self.state_update[key] = (local_state[key] - global_state[key].to(local_state[key].device)).cpu()  # * delta on the GPU, copied to the CPU once
        client_message = self.pack_other_message()  # * same metadata as the starter path
        client_message['update'] = self.state_update  # * attach the update to the message
        return client_message  # * message identical in format to the starter path

    def local_update(self, server_message: dict):
        if self.fast:  # * --gpu_fast path
            return self._local_update_fast(server_message)  # * dispatch to the fast path
        global_state = server_message['model_state']
        self.local_model.load_state_dict(global_state)

        optimizer = get_optimizer(self.args.client_optimizer, self.local_model.parameters(), self.args.client_lr, self.args.client_momentum, self.args.client_weight_decay)

        tic = time.time()
        self.local_model.cuda()
        self.local_model.train()
        for _ in range(self.args.client_n_epochs):
            for (inputs, targets) in self.dataloader:
                # drop batch of size 1 when there is batch normalization
                if self.args.architecture in ['resnet18', 'squeezenet']:
                    batch_size = len(inputs)
                    if batch_size == 1:
                        continue

                inputs, targets = inputs.cuda(), targets.cuda()
                
                optimizer.zero_grad()

                outputs = self.local_model(inputs)
                loss = self.loss(outputs, targets)
                
                loss.backward()
                if self.args.clip_max_norm is not None:
                    clip_grad_norm_(self.local_model.parameters(), max_norm=self.args.clip_max_norm)
                optimizer.step()
            self.local_model.cpu()
        
        with torch.no_grad():
            self.state_update = {}
            local_state = self.local_model.state_dict()
            for key in local_state:
                self.state_update[key] = local_state[key] - global_state[key]
        
        client_message = self.pack_other_message()
        client_message['update'] = self.state_update
        toc = time.time()

        return client_message
    
    def pack_other_message(self) -> dict:
        uploaded_message = {
            'n_data': len(self.data),  # type: ignore
        }

        return uploaded_message