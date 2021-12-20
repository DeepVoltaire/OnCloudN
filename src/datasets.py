import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import time


class LoadTifDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths,
        mask_paths,
        transforms=None,
        val=False,
        test=False,
    ):
        """Dataset for training, validating and testing S2 models.

        Args:
            img_paths (list of str): Paths to the input B02 path.
            mask_paths (list of str): Paths to the labels for the S2 images.
            transforms (albumentation.transforms, optional): Transforms to apply to the images/masks. Defaults to None.
            val (bool, optional): If True, this dataset is used for validation.
                Defaults to False.
            test (bool, optional): If True, we don't provide the label, because we are testing. Defaults to False.
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.val = val
        self.test = test

    def __getitem__(self, idx):
        sample = {}
        # Load in image
        arr_x = tifffile.imread([self.img_paths[idx].replace("B02.tif", f"B0{k}.tif") for k in [2, 3, 4, 8]])

        if arr_x.shape[0] < 20:
            arr_x = arr_x.transpose((1, 2, 0))

        arr_x = np.nan_to_num(arr_x)

        arr_x = (arr_x / 2 ** 16).astype(np.float32)

        sample["image"] = arr_x

        # Load in and preprocess label mask
        if not self.test:
            arr_y = tifffile.imread(self.mask_paths[idx]).squeeze()
            sample["mask"] = arr_y

        # Apply Data Augmentation
        if self.transforms:
            if sample["image"].shape[0] < 20:
                sample["image"] = sample["image"].transpose((1, 2, 0))
            sample = self.transforms(image=sample["image"], mask=sample["mask"])
        if sample["image"].shape[-1] < 20:
            sample["image"] = sample["image"].transpose((2, 0, 1))

        return sample

    def __len__(self):
        return len(self.img_paths)

    def visualize(self, how_many=1, show_specific_index=None):
        """Visualize a number of images from the dataset. The images are randomly selected unless show_specific_index is passed.

        Args:
            how_many (int, optional): number of images to visualize. Defaults to 1.
            show_specific_index (int, optional): If passed, only show the image corresponding to this index. Defaults to None.
        """
        for _ in range(how_many):
            rand_int = np.random.randint(len(self.img_paths))
            if show_specific_index is not None:
                rand_int = show_specific_index
            sample = self.__getitem__(rand_int)
            print(self.img_paths[rand_int], rand_int)
            print(self.mask_paths[rand_int])
            fig_cols = 3
            f, axarr = plt.subplots(1, fig_cols, figsize=(30, 12))

            img_string = "S2"
            arr_x = tifffile.imread([self.img_paths[rand_int].replace("B02.tif", f"B0{k}.tif") for k in [2, 3, 4]])
            if arr_x.shape[0] < 20:
                arr_x = arr_x.transpose((1, 2, 0))
            axarr[0].imshow(scale_S2_img(arr_x))

            img = sample["image"]
            axarr[0].set_title(f"{img_string} Image")  # . Min: {img.min():.4f}, Max: {img.max():.4f}")
            axarr[1].imshow(img[0] * 2 ** 16)
            axarr[1].set_title(f"Min: {img[:-1].min():.4f}, Max: {img[:-1].max():.4f}", fontsize=15)

            if "mask" in sample.keys():
                axarr[2].imshow(img[0] * 2 ** 16)
                mask = sample["mask"]
                print(f"Mask unique values: {np.unique(mask)}")
                axarr[2].set_title(f"Mask==1 px: {(mask == 1).sum()}", fontsize=15)
                axarr[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="spring", alpha=0.4)
            plt.tight_layout()
            plt.show()


def scale_S2_img(matrix, min_values=None, max_values=None):
    """Returns a scaled (H,W,D) image which is more easily visually inspectable. Image is linearly scaled between
    min and max_value of by channel"""
    w, h, d = matrix.shape
    min_values = np.array([100, 100, 100])
    max_values = np.array([2500, 2500, 2500])

    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    matrix = (matrix - min_values[None, :]) / (max_values[None, :] - min_values[None, :])
    matrix = np.reshape(matrix, [w, h, d])

    matrix = matrix.clip(0, 1)
    return matrix


class TestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths,
        transforms=None,
    ):
        """Dataset for predicting with S1/S2 models where each image loaded is automatically padded.

        Args:
            img_paths (list of list of str): Paths to the input S1 or S2 images.
            transforms (albumentation.transforms, optional): Transforms to apply to the images/masks. Defaults to None.
        """
        self.img_paths = img_paths
        self.transforms = transforms
        self.size_multiplier = 32

    def __getitem__(self, idx):
        # Load in image
        arr_x = np.nan_to_num(tifffile.imread(self.img_paths[idx]))

        if arr_x.shape[0] < 20:
            arr_x = arr_x.transpose((1, 2, 0))

        arr_x = (arr_x / 2 ** 16).astype(np.float32)

        sample = {"image": arr_x}

        # Apply Data Augmentation
        if self.transforms:
            if sample["image"].shape[0] < 20:
                sample["image"] = sample["image"].transpose((1, 2, 0))
            sample = self.transforms(image=sample["image"], mask=sample["mask"])
        if sample["image"].shape[-1] < 20:
            sample["image"] = sample["image"].transpose((2, 0, 1))
        # print(f"{self.mask_paths[idx]} - {sample['mask'].shape}, {self.img_paths[idx]} - {sample['image'].shape}")
        sample["path"] = self.img_paths[idx]
        return sample

    def __len__(self):
        return len(self.img_paths)
