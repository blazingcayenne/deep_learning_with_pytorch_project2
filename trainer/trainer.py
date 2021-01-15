"""
Unified class to make training pipeline for deep neural networks.
"""

import os
import datetime

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter

import torch

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .hooks import test_hook_default, train_hook_default
from .visualizer import Visualizer


class Trainer:  # pylint: disable=too-many-instance-attributes
    def __init__( # pylint: disable=too-many-arguments
        self,
        model: torch.nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        model_save_dir: Union[str, Path] = "checkpoints",
        model_name: str = "model",
        model_saving_period: int = 1,
        stop_loss_epochs: int = 0,
        stop_acc_ema_alpha: float = 0.1,
        stop_acc_epochs: int = 0,
        stop_acc_threshold: float = 1.0,
        device: Union[torch.device, str] = "cuda",
        data_getter: Callable = itemgetter("image"),
        target_getter: Callable = itemgetter("target"),
        stage_progress: bool = True,
        visualizer: Union[Visualizer, None] = None,
        get_key_metric: Callable = itemgetter("top1")
    ):
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_save_dir = model_save_dir
        self.model_name = model_name
        self.model_saving_period = model_saving_period
        self.stop_loss_epochs = stop_loss_epochs
        self.stop_acc_ema_alpha = stop_acc_ema_alpha
        self.stop_acc_epochs = stop_acc_epochs
        self.stop_acc_threshold = stop_acc_threshold
        self.device = device
        self.data_getter = data_getter
        self.target_getter = target_getter
        self.stage_progress = stage_progress
        self.visualizer = visualizer
        self.get_key_metric = get_key_metric
        self.hooks = {}
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "test_metric": []}
        self._register_default_hooks()

    def fit(self, epochs):
        """ Fit model method.

        Arguments:
            epochs (int): number of epochs to train model.
        """
        lowest_loss = float("inf")
        smoothed_accuracy_history = []

        epochs_since_lowest_loss = 0
        os.makedirs(self.model_save_dir, exist_ok=True)
        iterator = tqdm(range(epochs), dynamic_ncols=True)
        for epoch in iterator:
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter
            )
            output_test = self.hooks["test"](
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric
            )
            if self.visualizer:
                self.visualizer.update_charts(
                    None, output_train['loss'], output_test['metric'], output_test['loss'],
                    self.optimizer.param_groups[0]['lr'], epoch
                )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()

            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train, output_test)

            # track of the lowest loss and the number of epochs since the lowest loss
            epochs_since_lowest_loss += 1
            current_loss = output_test['loss']
            if current_loss < lowest_loss:
                lowest_loss = current_loss
                epochs_since_lowest_loss = 0

            # model_saving_period (msp)
            if self.model_saving_period > 0:
                # when msp > 0, save the model every msp epoch(s)
                if (epoch + 1) % self.model_saving_period == 0:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.model_save_dir, 
                            self.model_name) + str(datetime.datetime.now() + ".pt"
                        )
                    )
            elif self.model_saving_period == 0 and epochs_since_lowest_loss == 0:
                # when msp == 0, save the model when the test loss is a new low
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_save_dir, self.model_name + ".pt")
                )

            # if enabled, terminate training loop after specified number of epochs without
            # test loss reduction
            if self.stop_loss_epochs > 0:
                if epochs_since_lowest_loss >= self.stop_loss_epochs:
                    break
                
            # if enabled, terminate training loop after specified number of epochs without
            # significant increase in (smoothed) accuracy
            if self.stop_acc_epochs > 0:
                # ToDo: Pass a method to retrieve the accuracy into this fit method.
                #       The current implementation depends upon the passed metric function!
                current_accuracy = output_test['metric']['top1']

                # compute smoothed accuracy
                if (len(smoothed_accuracy_history) == 0):
                    smoothed_accuracy = current_accuracy    
                else:
                    smoothed_accuracy = self.stop_acc_ema_alpha * current_accuracy + \
                        (1 - self.stop_acc_ema_alpha) * smoothed_accuracy_history[-1]

                # check accuracy progress
                if (len(smoothed_accuracy_history) >= self.stop_acc_epochs):
                    delta_accuracy = smoothed_accuracy - smoothed_accuracy_history[-self.stop_acc_epochs]
                    if delta_accuracy <= self.stop_acc_threshold:
                        break;

                # store smoothed accuracy
                smoothed_accuracy_history.append(smoothed_accuracy)
        
        return self.metrics

    def register_hook(self, hook_type, hook_fn):
        """ Register hook method.

        Arguments:
            hook_type (string): hook type.
            hook_fn (callable): hook function.
        """
        self.hooks[hook_type] = hook_fn

    def _register_default_hooks(self):
        self.register_hook("train", train_hook_default)
        self.register_hook("test", test_hook_default)
        self.register_hook("end_epoch", None)
