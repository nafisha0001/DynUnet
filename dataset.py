import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
from utils.utils import resample_segmentation_to_image

reader = sitk.ImageSeriesReader()

class VSDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, target_slices=None):
        self.data = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
        self.target_slices = target_slices
        self.image_filenames = self.data['image_path'].tolist()
        self.mask_filenames = self.data['SegmentationPath'].tolist()

    def transform_volume(self, image_volume, mask_volume):
        image_volume = image_volume.transpose(2, 1, 0)  
        mask_volume = mask_volume.transpose(2, 1, 0)   
        # print('before:-', image_volume.shape, mask_volume.shape)
        transformed = self.transform(
            image=image_volume.astype(np.float32),  
            mask=mask_volume.astype(np.float32)
        )
        images = transformed['image']
        masks = transformed['mask']
        
        images = images.permute(0, 2, 1)
        masks= masks.permute(2, 1, 0)

        # print('after:- ',images.shape, masks.shape)
        return images, masks.float()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.data_dir, self.mask_filenames[idx])

        dicom_series = reader.GetGDCMSeriesFileNames(image_path)
        reader.SetFileNames(dicom_series)
        reference_image = reader.Execute()

        segmentation_image = sitk.ReadImage(mask_path)
        segmentation_resampled = resample_segmentation_to_image(segmentation_image, reference_image)

        image= sitk.GetArrayFromImage(reference_image)
        mask= sitk.GetArrayFromImage(segmentation_resampled)

        if self.transform:
            transformed_image_volume, transformed_mask_volume = self.transform_volume(image, mask)
        
        # Change to (C, D, H, W)
        image_tensor = transformed_image_volume.unsqueeze(0)  
        mask_tensor = transformed_mask_volume.unsqueeze(0)
        
        return image_tensor, mask_tensor