import os
import torch
import numpy as np
import torch.nn as nn

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def make_mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes)-2:
            # layers += [nn.Linear(sizes[j], sizes[j+1]), nn.BatchNorm1d(sizes[j+1]), activation()]
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
        else:  # output layer
            layers += [nn.Linear(sizes[j], sizes[j+1]), output_activation()]
    return nn.Sequential(*layers)


def huber_loss(x, delta=10.):
    """
    Compute the huber loss.
    Ref: https://en.wikipedia.org/wiki/Huber_loss
    """

    delta = torch.ones_like(x) * delta
    less_than_max = 0.5 * (x * x)
    greater_than_max = delta * (torch.abs(x) - 0.5 * delta)

    return torch.where(
        torch.abs(x) <= delta,
        less_than_max,
        greater_than_max
    )


def get_default_rb_dict(obs_dim, act_dim, size):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": obs_dim
            },
            "act": {
                "shape": act_dim
            },
            "rew": {},
            "next_obs": {
                "shape": obs_dim
            },
            "done": {},
        }
    }


class Logger:
    def __init__(self):
        self.start_date = datetime.now().strftime("%b_%d_%Y_%H%M%S")
        self.steps = 0
        self.setup_directory()
        self.writer = SummaryWriter(f'./save/{self.start_date}/')

    def setup_directory(self):
        if not os.path.exists(f'./save/{self.start_date}/models'):
            os.makedirs(f'./save/{self.start_date}/models')

    def update_steps(self):
        self.steps += 1

    def save_model(self, model):
        torch.save(model, f'./save/{self.start_date}/models/sac_{self.steps + 1}.pth')

    def store(self, name, val):
        self.writer.add_scalar(name, val, self.steps)
