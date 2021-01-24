from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

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
class DataAugConfig:
    # enabled color augmentation
    color_enabled: bool = True

    # maximum amount to jitter brightness
    color_brightness: Tuple[float, float] = (0.85, 1.15)

    # maximum amount to jitter contrast
    color_contrast: Tuple[float, float] = (0.5, 1.5)

    # maximum amount to jitter saturation
    color_saturation: Tuple[float, float] = (0.5, 2.0)

    # maximum amount to jitter hue
    color_hue: Tuple[float, float] = (-0.03, 0.03)

    # probability of horizontally flipping
    horz_flip_prob: float = 0.5

    # probability of vertically flipping
    vert_flip_prob: float = 0.5

    # enabled affine augmentation
    affine_enabled: bool = True
    
    # maximum amount to rotate in degrees
    affine_rotation: float = 45

    # maximum amount to horizontally and vertically translate
    affine_translate: Tuple[float, float] = (0.1, 0.1)

    # maximum amount of shear to apply
    affine_shear: Tuple[float, float] = (-0.1, 0.1)

    # scaling range
    affine_scale: Tuple[float, float] = (0.9, 1.1)

    # probability erasing will be performed
    erasing_prob: float = 0.5

    # range of proportion of erased area against input image
    erasing_scale: Tuple[float, float] = (0.02, 0.33)

    # range of aspect ratio of erased area
    erasing_ratio: Tuple[float, float] = (0.3, 3.3)

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

    # data transformations to use during test or validation data preparation
    # (also used during pipeline check training)
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )

    #data transformations to use during visualization
    visual_transforms: Iterable[Callable] = (
        ToTensor(),
    )

    #data transformations to use during visualization of augmented data
    visual_aug_transforms: Iterable[Callable] = (
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
    learning_rate: float = 0.0005

    # SGD optimizer - adds robustness with local minimas
    momentum: float = 0.9

    # SGD optimizer - amount of additional regularization on the weights values
    weight_decay: float = 0.0001

    # Adam optimizer - running average coefficients
    betas: Tuple[float, float] = (0.9, 0.999)

@dataclass
class SchedulerConfig:
    # multiplicative factor of learning rate decay
    gamma: float = 0.5

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

    # model save periodicity (> 0), save when test loss lowers (= 0), disable (< 0)
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
