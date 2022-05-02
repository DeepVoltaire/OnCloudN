import attr
import cv2
import yaml
import os
import pprint
import glob
from attr.validators import in_


@attr.s
class HyperParams(object):
    """Class for easily transporting hyperparameters"""

    #########################################################################
    ### Data
    #########################################################################

    df_path = attr.ib(default="/data/c2s-benchmark-dataset/catalogues/splits/xxx.pkl", type=str)

    ## Data Setup
    fold_nb = attr.ib(default=0, type=int)
    only_val = attr.ib(default=False, type=bool)
    train_batch_size = attr.ib(default=32, type=int)
    val_batch_size = attr.ib(default=32, type=int)

    ## Data Augmentation on CPU
    train_crop_size = attr.ib(default=256, type=int)
    randomsizecrop_interpolation = attr.ib(default=cv2.INTER_LINEAR)
    da_p_cutout = attr.ib(default=0.0, type=float, converter=float)
    cutmix_alpha = attr.ib(default=0.0, type=float, converter=float)
    random_sized_params = attr.ib(default=[0.0, 0.0])
    da_brightness_magnitude = attr.ib(default=0.0, type=float, converter=float)
    da_contrast_magnitude = attr.ib(default=0.0, type=float, converter=float)

    #########################################################################
    ### Training
    #########################################################################

    ## Experiment Setup
    name = attr.ib(default="test_name", type=str)

    ## Model
    num_classes = attr.ib(default=1, type=int)
    input_channel = attr.ib(default=2, type=int)
    extra_bands = attr.ib(default=[])
    smp_decoder_use_batchnorm = attr.ib(default=True, type=bool)
    smp_decoder_channels_mult = attr.ib(default=1.0, type=float, converter=float)
    smp_decoder_use_attention = attr.ib(default=False, type=bool)
    backbone = attr.ib(default="timm_efficientnet_b1", type=str)
    pretrained = attr.ib(default=True, type=bool)
    model = attr.ib(
        default="unet",
        validator=in_(
            ["unet", "unetplusplus"],
        ),
    )

    ## Training Setup
    resume = attr.ib(default="", type=str)
    n_epochs = attr.ib(default=1000, type=int)
    use_fp16 = attr.ib(default=True, type=bool)
    patience = attr.ib(default=4, type=int)
    num_batches = attr.ib(default=0, type=int)
    print_freq = attr.ib(default=1000, type=int)

    ### Optimizer
    weight_decay = attr.ib(default=0.0, type=float, converter=float)
    lr = attr.ib(default=1e-3, type=float, converter=float)

    ### Data Augmentation on GPU
    gpu_da_params = attr.ib(default=[0.25])

    ### Loss, Metric
    loss = attr.ib(
        default="xedice",
        validator=in_(
            ["xedice", "lovasz", "focal"],
        ),
    )
    alpha = attr.ib(default=0.5, type=float, converter=float)

    def __str__(self):
        return pprint.pformat(attr.asdict(self))


def open_from_yaml(yaml_path):
    if not yaml_path.endswith(".yaml"):
        yaml_paths = glob.glob(f"{yaml_path}/fold_0/*/*.yaml")
        if len(yaml_paths) == 0:
            raise ValueError(f"Tried to look for a .yaml file in {yaml_path}/fold_0/*/*.yaml, but found none")
        else:
            yaml_path = yaml_paths[0]
    with open(yaml_path) as file:
        documents = yaml.full_load(file)
    return documents


def save_hps_as_yaml(hps, yaml_save_path):
    os.makedirs("/".join(yaml_save_path.split("/")[:-1]), exist_ok=True)
    with open(yaml_save_path, "w") as file:
        yaml.dump(attr.asdict(hps), file)
