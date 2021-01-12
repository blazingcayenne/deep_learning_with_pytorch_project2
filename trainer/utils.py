# # <font style="color:blue">Utils</font>
#
# Implements helper functions.

import random

import numpy as np
import torch

from .configuration import SystemConfig, TrainerConfig, DataLoaderConfig


# ## <font style="color:green">AverageMeter</font>
#
# Computes and stores the average and current value.

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


# ## <font style="color:green">Patch Configs</font>
#
# Patches configs if cuda is not available

def patch_configs(epoch_num_to_set=TrainerConfig.training_epochs, batch_size_to_set=DataLoaderConfig.batch_size):
    """ Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataLoaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataLoaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, training_epochs=epoch_num_to_set)
    return dataloader_config, trainer_config


# ## <font style="color:green">Setup System</font>

def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic
