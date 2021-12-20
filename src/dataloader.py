import torch
import pandas as pd
import albumentations
from .datasets import LoadTifDataset


def get_dataloaders(hps):
    """Builds datasets and dataloaders for training/validation"""
    df = pd.read_pickle(hps.df_path)
    num_workers = 8

    ## Setup the correct image and label paths
    df["img_path"] = df["B02_path"].copy()

    img_paths_val = df.loc[df["fold"] == hps.fold_nb, "img_path"].tolist()
    mask_paths_val = df.loc[df["fold"] == hps.fold_nb, "label_path"].tolist()
    val, test = True, False

    val_dataset = LoadTifDataset(
        img_paths_val,
        mask_paths_val,
        val=val,
        test=test,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hps.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    if hps.only_val:
        return val_dataset, val_loader

    ## Building the training dataset and dataloader
    # Data Augmentations
    train_transforms = get_train_transforms(hps)
    print(f"Train Data Augmentations: {train_transforms}")
    print("#" * 100)

    # img_paths_train, mask_paths_train, weights_all =  [], [], []

    # total_weight_gt = df[(df["fold"] != hps.fold_nb) & (df["batch"] == "sen1floods11_gt_rechip")].shape[0]

    # for batch, batch_name in zip(BATCH_DF_NAMES, BATCH_NAMES):
    #     df_batch = df[(df["fold"] != hps.fold_nb) & (df["batch"] == batch)].copy()
    #     if not len(df_batch):
    #         warnings.warn(f"No training examples for batch {batch}, skipping")
    #         continue

    #     img_paths_batch = df_batch["img_path"].tolist()
    #     mask_paths_batch = df_batch["label_path"].tolist()

    #     df_batch["weights"] = 1
    #     gt_to_batch_weight_ratio = get_weight_ratio(hps, batch)
    #     if gt_to_batch_weight_ratio == 0:  # skip this batch
    #         continue

    #     weights_batch = (
    #         df_batch["weights"] * (1 / gt_to_batch_weight_ratio) * (total_weight_gt / len(img_paths_batch))
    #     ).tolist()
    #     weights_batches[batch_name] = weights_batch
    #     weights_all.extend(weights_batches[batch_name])

    #     total_weight_batch = sum(weights_batch)
    #     gt_to_batch_ratio_real = total_weight_gt / total_weight_batch

    #     img_paths_train.extend(img_paths_batch)
    #     gsw_paths_train.extend(df_batch["gsw_label_path"].tolist())
    #     mask_paths_train.extend(mask_paths_batch)
    #     biomes_train.extend(df_batch["biome_reduced"].tolist())
    #     dem_paths_train.extend(df_batch["dem_path"].tolist())
    #     lcc_paths_train.extend(df_batch["lcc_path"].tolist())

    #     if not np.isclose(gt_to_batch_ratio_real, gt_to_batch_weight_ratio):
    #         raise ValueError(
    #             f"{batch_name} is {gt_to_batch_ratio_real:.2f} and should be {gt_to_batch_weight_ratio:.2f}"
    #         )
    #     print(
    #         f"Total {batch_name:25} weight: {sum(weights_batch):6.1f} for {len(weights_batch):6} images "
    #         f"({sum(weights_batch) / len(weights_batch):.3f} average weight)"
    #     )
    img_paths_train = df.loc[df["fold"] != hps.fold_nb, "img_path"].tolist()
    mask_paths_train = df.loc[df["fold"] != hps.fold_nb, "label_path"].tolist()
    weights_all = [1] * len(img_paths_train)  # np.full(len(img_paths_train), 1)

    print("#" * 100)
    print(f"Fold {hps.fold_nb} --> Train: {len(img_paths_train)}, Val: {len(val_dataset)}")
    print("#" * 100)

    # repeat all paths if we need longer train epochs
    while hps.num_batches > int(len(img_paths_train) / hps.train_batch_size):
        print(
            f"We want to train {hps.num_batches} batches in each epoch, but our current dataset is only "
            f"{int(len(img_paths_train) / hps.train_batch_size)} batches long. Doubling dataset size"
        )
        img_paths_train = img_paths_train * 2
        mask_paths_train = mask_paths_train * 2
        weights_all = weights_all * 2

    train_dataset = LoadTifDataset(
        img_paths_train,
        mask_paths_train,
        transforms=albumentations.Compose(train_transforms),
    )

    # weights = [1] * len(weights_all) # for debugging
    weights_all = torch.Tensor(weights_all)
    weights_all = weights_all.double()
    msg = (
        f"Training dataset and sampling weights have not the same length: {len(train_dataset)} and {len(weights_all)}"
    )
    assert len(train_dataset) == len(weights_all), msg
    if sum(weights_all) == len(weights_all):
        sampler, train_shuffle = None, True
    else:
        sampler, train_shuffle = (
            torch.utils.data.sampler.WeightedRandomSampler(weights_all, len(weights_all)),
            False,
        )
        print("Using a Weighted Random Sampler with different weights for different batches of data")
        print("#" * 100)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hps.train_batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_dataset, train_loader, val_dataset, val_loader


def get_train_transforms(hps):
    """Returns train transforms using information from hps"""
    train_transforms = []
    if hps.random_sized_params[0]:
        train_transforms.append(
            albumentations.RandomSizedCrop(  # default params [0.875, 1.125]
                min_max_height=(
                    int(hps.train_crop_size * hps.random_sized_params[0]),
                    int(hps.train_crop_size * hps.random_sized_params[1]),
                ),
                height=hps.train_crop_size,
                width=hps.train_crop_size,
                interpolation=hps.randomsizecrop_interpolation,
            )
        )  # cv2.INTER_NEAREST faster than cv2.INTER_LINEAR faster than cv2.INTER_CUBIC
    else:
        train_transforms.append(albumentations.RandomCrop(hps.train_crop_size, hps.train_crop_size))
    if hps.da_p_cutout:
        train_transforms.append(
            albumentations.Cutout(
                num_holes=8,
                max_h_size=int(hps.train_crop_size * 0.1875),
                max_w_size=int(hps.train_crop_size * 0.1875),
                p=hps.da_p_cutout,
            )
        )
    return train_transforms
