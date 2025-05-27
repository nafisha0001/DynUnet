import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
from utils.utils import resample_segmentation_to_image, load_spacing, resample_pair

reader = sitk.ImageSeriesReader()

class VSDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
        self.image_filenames = self.data['image_path'].tolist()
        self.mask_filenames = self.data['SegmentationPath'].tolist()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.data_dir, self.mask_filenames[idx])

        # Read DICOM image series
        dicom_series = reader.GetGDCMSeriesFileNames(image_path)
        reader.SetFileNames(dicom_series)
        reference_image = reader.Execute()

        # Read and resample mask
        segmentation_image = sitk.ReadImage(mask_path)
        segmentation_resampled = resample_segmentation_to_image(segmentation_image, reference_image)

        # Convert to numpy arrays
        image = sitk.GetArrayFromImage(reference_image).astype(np.float32)  # (D, H, W)
        mask = sitk.GetArrayFromImage(segmentation_resampled).astype(np.uint8)

        # Resample to fixed spacing
        spacing = load_spacing(reference_image)
        image, mask = resample_pair(image, mask, spacing)

        # Add channel dimension to image and mask → (C=1, D, H, W)
        sample = {
            "image": image[np.newaxis, ...],  # (1, D, H, W)
            "mask": mask[np.newaxis, ...]     # (1, D, H, W)
        }

        # print(f"Image shape: {sample['image'].shape}, Mask shape: {sample['mask'].shape}")

        # Apply 3D transform
        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["mask"]