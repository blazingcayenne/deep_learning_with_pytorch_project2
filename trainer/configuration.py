from typing import Callable, Iterable, Tuple
from dataclasses import dataclass

from torchvision.transforms import ToTensor


@dataclass
class SystemConfig:
    # project directory
    proj_dir: str = "."

    # random number generator seed
    seed: int = 42

    # make cudnn deterministic (reproducible training)
    cudnn_deterministic: bool = True

    # enable CuDNN benchmark for the sake of performance
    cudnn_benchmark_enabled: bool = False


@dataclass
class DatasetConfig:
    # dataset directory relative to project directory
    data_dir: str = "data"

    # percent of training data to use for validation
    valid_size: float = 0.2

    # data transformations to use during training data preparation
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )

    # data transformations to use during test data preparation
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )

    #data transformations to use during data visualization
    visual_transforms: Iterable[Callable] = (
        ToTensor(),
    )


@dataclass
class DataLoaderConfig:
    # amount of data to pass through the network at each f-b iteration
    batch_size: int = 32

    # number of concurrent processes using to prepare data
    num_workers: int = 4


@dataclass
class OptimizerConfig:
    # determines the speed of network's weights update
    learning_rate: float = 0.001

    # SGD optimizer - adds robustness with local minimas
    momentum: float = 0.9

    # SGD optimizer - amount of additional regularization on the weights values
    weight_decay: float = 0.0001

    # Adam optimizer - running average coefficients
    betas: Tuple[float, float] = (0.9, 0.999)

@dataclass
class SchedulerConfig:
    # multiplicative factor of learning rate decay
    gamma: float = 0.1

    # period of learning rate decay
    step_size: int = 10
        
    # list of epoch monotonically increasing indices in which to decay learning rate
    milestones: Iterable = (20, 30, 40)

    # number of epochs with no improvement after which learning rate will be reduced
    patience: int = 10
        
    # threshold for measuring the new optimum, to only focus on significant changes.     
    threshold: float = 0.0001


@dataclass
class TrainerConfig:
    # device to use for training
    device: str = "cuda"

    # number of training iterations
    training_epochs: int = 50

    # enable progress bar visualization during train process
    progress_bar: bool = True

    # directory to save model state(s) relative to project directory
    model_dir: str = "models"

    # model save periodicity (> 0) or save when test loss lowers (<= 0)
    model_saving_period: int = 0

    # directory in which to save visualizations relative to project directory
    visualizer_dir: str = "runs"

    # stop training loss - stop after this many epochs w/o lowering test loss
    stop_loss_epochs: int = 0

    # stop training accuracy - stop after this many epochs w/o significant accuracy increase
    stop_acc_epochs: int = 0

    # stop training accuracy - exponential moving average alpha factor to smooth accuracy
    stop_acc_ema_alpha: float = 0.3

    # stop training accuracy - defines what constitutes significant increase in accuracy
    stop_acc_threshold: float = 2.0
