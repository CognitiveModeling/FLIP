import os
from utils.configuration import Configuration
import time
import torch as th
import numpy as np
from einops import rearrange, repeat, reduce
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule

class PeriodicCheckpoint(Callback):
    def __init__(self, save_path, save_every_n_steps = 3000):
        super().__init__()
        self.save_path = save_path
        self.save_every_n_steps = save_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if (global_step + 1) % self.save_every_n_steps == 0:
            checkpoint_file = os.path.join(self.save_path, "quicksave.ckpt")
            print(f"Saving checkpoint to {checkpoint_file}")
            trainer.save_checkpoint(checkpoint_file)

class Timer:
    
    def __init__(self):
        self.last   = time.time()
        self.passed = 0
        self.sum    = 0

    def __str__(self):
        self.passed = self.passed * 0.99 + time.time() - self.last
        self.sum    = self.sum * 0.99 + 1
        passed      = self.passed / self.sum
        self.last = time.time()

        if passed > 1:
            return f"{passed:.2f}s/it"

        return f"{1.0/passed:.2f}it/s"

class AverageDict:
    def __init__(self, memory = -1):
        # Store each key's (sum, count) in a regular dict
        self._storage = {}
        self.decay = np.exp(-1 / memory) if memory > 0 else 1

    def __getitem__(self, key):
        if key not in self._storage:
            return 0.0
        s, c = self._storage[key]
        if c == 0:
            return 0.0
        return s / c

    def __contains__(self, key):
        return key in self._storage

    def update(self, key, increment):
        if key not in self._storage:
            self._storage[key] = [increment, 1]
        else:
            self._storage[key][0] = self._storage[key][0] * self.decay + increment
            self._storage[key][1] = self._storage[key][1] * self.decay + 1

    def __repr__(self):
        rep = "AverageDict:\n"
        for k, (v, c) in self._storage.items():
            rep += f"  {k}: {v} / {c} = {v / c}\n"

        return rep
            

