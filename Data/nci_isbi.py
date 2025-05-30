import os
import pandas as pd
import torch
from PIL import Image
import numpy as np
from .base_dataset import BaseImageDataset
import pydicom as dicom
from torchvision import transforms
from monai.transforms import (
    MapLabelValue,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Resized,
    RandFlipd,
    ToTensord,
    Compose,
)

class NCIISBIProstateDataset(BaseImageDataset):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)

        split_file = pd.read_csv(os.path.join(self.root, f"{self.split}.txt"), sep=" ", header=None)
        self.sample_list = split_file.iloc[:, 0].tolist()

        self.images = np.array([os.path.join(self.root, "images", f"{file}.dcm") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]
        self.labels = np.array([os.path.join(self.root, "masks", f"{file}.png") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]

    def _fetch_data(self, idx):
        image = dicom.dcmread(self.images[idx]).pixel_array.astype(np.float32)
        label = np.array(Image.open(self.labels[idx]).convert("L"), dtype=np.int16)

        return {"image": image, "label": label}

    def _transform_custom(self, data):
        data["label"] = (data["label"].float() / 255.0).long()

        return data

    def get_transform(self):
        resize_keys = ["image", "label"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest"] if self.resize_label else ["bilinear"]

        if self.split == "train" and self.do_augmentation:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )


class NCIISBIProstateGazeDataset(NCIISBIProstateDataset):
    def __init__(self, pseudo_mask_root, *argv, **kargs):
        super().__init__(*argv, **kargs)

        self.pseudo_mask_root = pseudo_mask_root

        self.pseudo_labels = [np.array([os.path.join(pseudo_mask_root, f"heatmap", f"{file}.png",) for file in self.sample_list])]

    def _transform_custom(self, data):
        data = super()._transform_custom(data)
        return data

    def get_transform(self):
        resize_keys = ["image", "label", "pseudo_label"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest", "bilinear"] if self.resize_label else ["bilinear"]

        if self.split == "train" and self.do_augmentation:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label", "pseudo_label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    RandFlipd(
                        keys=["image", "label", "pseudo_label"],
                        prob=0.5,
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        keys=["image", "label", "pseudo_label"],
                        prob=0.5,
                        spatial_axis=1,
                    ),
                    ToTensord(keys=["image", "label", "pseudo_label"]),
                ]
            )
        else:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label", "pseudo_label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image", "pseudo_label"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    ToTensord(keys=["image", "label", "pseudo_label"]),
                ]
            )

    def _fetch_data(self, idx):
        data = super()._fetch_data(idx)
        pseudo_label = np.array(Image.open(self.pseudo_labels[0][idx])).astype(np.float32)
        data[f"pseudo_label"] = pseudo_label / 255
        return data

def visualize(data, path):
    data = (data - data.min()) / (data.max() - data.min())
    data = np.array(data[0])
    data = Image.fromarray((data * 255).astype(np.uint8))
    data.save(path)

if __name__ == "__main__":
    dataset = NCIISBIProstateGazeDataset(
                root="NCI-ISBI-2013/",
                pseudo_mask_root=os.path.join("NCI-ISBI-2013/", "gaze"),
                split='train',
                spatial_size=224,
                do_augmentation=True,
                resize_label=True,
                size_rate=1,
            )
    data = dataset[0]
    print(data["image"].shape, data["label"].shape, data["pseudo_label"].shape)

    dataset = NCIISBIProstateDataset(
        root="NCI-ISBI-2013/",
        split='test',
        spatial_size=224,
        do_augmentation=True,
        resize_label=False,
        size_rate=1,
    )
    data = dataset[0]
    print(data["image"].shape, data["label"].shape)





























