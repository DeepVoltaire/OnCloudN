import os
from pathlib import Path
from typing import List

from loguru import logger
import pandas as pd
from PIL import Image
import torch
import typer
import numpy as np
import tifffile
import segmentation_models_pytorch as smp

#### DRIVENDATA
ROOT_DIRECTORY = Path("/codeexecution")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"

# ###### LOCAL TESTING
# ROOT_DIRECTORY = Path("benchmark_src")
# DATA_DIRECTORY = Path("data")
# PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
# ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
# INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"

# Set the pytorch cache directory and include cached models in your submission.zip
os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "assets/torch")

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __getitem__(self, idx):
        arr_x = tifffile.imread([self.img_paths[idx].replace("B02.tif", f"B0{k}.tif") for k in [2, 3, 4, 8]])
        arr_x = np.nan_to_num(arr_x)
        arr_x = (arr_x / 2 ** 16).astype(np.float32)

        sample = {"image": arr_x}
        sample["chip_id"] = self.img_paths[idx].split("/")[-2]
        return sample

    def __len__(self):
        return len(self.img_paths)

def get_metadata(features_dir, bands):
    """
    Given a folder of feature data, return a dataframe where the index is the chip id
    and there is a column for the path to each band's TIF image.

    Args:
        features_dir (os.PathLike): path to the directory of feature data, which should have
            a folder for each chip
        bands (list[str]): list of bands provided for each chip
    """
    chip_metadata = pd.DataFrame(index=[f"{band}_path" for band in bands])
    chip_ids = (
        pth.name for pth in features_dir.iterdir() if not pth.name.startswith(".")
    )

    for chip_id in chip_ids:
        chip_bands = [f"{features_dir}/{chip_id}/{band}.tif" for band in bands]
        chip_metadata[chip_id] = chip_bands

    return chip_metadata.transpose().reset_index().rename(columns={"index": "chip_id"})


def make_predictions(models, x_paths, bands, predictions_dir):
    """Predicts cloud cover and saves results to the predictions directory.

    Args:
        model (CloudModel): an instantiated CloudModel based on pl.LightningModule
        x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands provided
        bands (list[str]): list of bands provided for each chip
        predictions_dir (os.PathLike): Destination directory to save the predicted TIF masks
    """
#     print(x_paths)
    test_dataset = TestDataset(img_paths=x_paths)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=14,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
    torch.set_grad_enabled(False)
    for batch_index, batch in enumerate(test_dataloader):
        logger.debug(f"Predicting batch {batch_index} of {len(test_dataloader)}")
        x = batch["image"].cuda(non_blocking=True)
        preds = torch.softmax(models[0](x), dim=1)[:, 1]
        for model_nb in range(1, len(models)):
            preds += torch.softmax(models[model_nb](x), dim=1)[:, 1]
        preds /= len(models)
        preds = (preds > 0.45).cpu().numpy().astype("uint8")
        for chip_id, pred in zip(batch["chip_id"], preds):
            chip_pred_path = predictions_dir / f"{chip_id}.tif"
            chip_pred_im = Image.fromarray(pred)
            chip_pred_im.save(chip_pred_path)


def main(
    experiment_names=["Exp4"] * 5  + ["Exp8_3"] * 5 + ["Exp11-0"] * 5 + ["Exp7-3"] * 5, # + ["Exp4-1"] * 5,
    fold_list=[0, 1, 2, 3, 4] * 4,
#     experiment_names=["Exp10-0"] * 5,
#     fold_list=[0, 1, 2, 3, 4] * 1,
    trained_models_dir: Path = ASSETS_DIRECTORY,
    test_features_dir: Path = DATA_DIRECTORY / "test_features",
    predictions_dir: Path = PREDICTIONS_DIRECTORY,
    bands: List[str] = ["B02", "B03", "B04", "B08"],
):
    """
    Generate predictions for the chips in test_features_dir using the model saved at
    model_weights_path.

    Predictions are saved in predictions_dir. The default paths to all three files are based on
    the structure of the code execution runtime.

    Args:
        hps_paths: Path to the model hyperparameters.
        test_features_dir (os.PathLike, optional): Path to the features for the test data. Defaults
            to 'data/test_features' in the same directory as main.py
        predictions_dir (os.PathLike, optional): Destination directory to save the predicted TIF masks
            Defaults to 'predictions' in the same directory as main.py
        bands (List[str], optional): List of bands provided for each chip
    """
    if not os.path.exists(test_features_dir):
        raise ValueError(
            f"The directory for test feature images must exist and {test_features_dir} does not exist"
        )
    os.makedirs(predictions_dir, exist_ok=True)

    models = []
    for experiment_name, fold_nb in zip(experiment_names, fold_list):
        jit_model_path = f"{trained_models_dir}/CNN-{experiment_name}_fold{fold_nb}_jit.pt"
        if not os.path.exists(jit_model_path):
            raise ValueError(f"Model path {jit_model_path} not found.")
        model = torch.jit.load(jit_model_path)
        model.eval()
        model = model.cuda()
        models.append(model)
        logger.info(f"Using {jit_model_path}")
    
    logger.info("Loading test metadata")
    test_metadata = get_metadata(test_features_dir, bands=bands)
    logger.info(f"Found {len(test_metadata)} chips")

    logger.info("Generating predictions in batches")
    make_predictions(models, test_metadata["B02_path"].tolist(), bands, predictions_dir)
    logger.info(f"""Saved {len(list(predictions_dir.glob("*.tif")))} predictions""")

if __name__ == "__main__":
    typer.run(main)
