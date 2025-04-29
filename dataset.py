import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.utils import load_nifti_as_dhw, dicom_load

class VSDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, target_slices=None):
        self.data = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
        self.target_slices = target_slices
        self.image_filenames = self.data['image_path'].tolist()
        self.mask_filenames = self.data['SegmentationPath'].tolist()

    def transform_volume(self, image_volume, mask_volume):
        image_volume = image_volume.transpose(1, 2, 0)  
        mask_volume = mask_volume.transpose(1, 2, 0)   
        # print('before:-', image_volume.shape, mask_volume.shape)
        transformed = self.transform(
            image=image_volume,
            mask=mask_volume
        )
        images = transformed['image']
        masks = transformed['mask']
        masks= masks.permute(2, 0, 1)
        # print('after:- ',images.shape, masks.shape)
        return images, masks.float()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.data_dir, self.mask_filenames[idx])

        mask = load_nifti_as_dhw(mask_path)
        image = dicom_load(image_path, mask.shape)
        # print(image.shape, mask.shape)

        transformed_image_volume, transformed_mask_volume = self.transform_volume(image, mask)
        
        # # Change to (C, D, H, W)
        image_tensor = transformed_image_volume.unsqueeze(0)  
        mask_tensor = transformed_mask_volume.unsqueeze(0)   
        # print(image_tensor.shape, mask_tensor.shape)

        # Now handle expansion or removal
        current_slices = image_tensor.shape[1]

        if self.target_slices:
            if current_slices < self.target_slices:
                # --- Duplicate labeled slices ---
                # required_slices = self.target_slices - current_slices
                labeled_slices = []
                for i in range(current_slices):
                    unique_vals = torch.unique(mask_tensor[0, i, :, :])
                    if 0 in unique_vals and 1 in unique_vals:
                        labeled_slices.append(i)

                if len(labeled_slices) == 0:
                    raise ValueError("No labeled slices (with both 0 and 1) found. Cannot duplicate.")

                while current_slices < self.target_slices:
                    for i in labeled_slices:
                        if current_slices < self.target_slices:
                            image_tensor = torch.cat((image_tensor, image_tensor[:, i:i+1, :, :]), dim=1)
                            mask_tensor = torch.cat((mask_tensor, mask_tensor[:, i:i+1, :, :]), dim=1)
                            current_slices += 1

            elif current_slices > self.target_slices:
                # --- Remove unlabeled slices ---
                unlabeled_slices = []
                for i in range(current_slices):
                    unique_vals = torch.unique(mask_tensor[0, i, :, :])
                    if torch.all(unique_vals == 0):
                        unlabeled_slices.append(i)

                if len(unlabeled_slices) == 0:
                    raise ValueError("No unlabeled slices found for removal.")

                slices_to_keep = list(range(current_slices))

                # Remove unlabeled slices until reaching target
                for i in unlabeled_slices:
                    if len(slices_to_keep) > self.target_slices:
                        slices_to_keep.remove(i)

                # After removal, crop
                image_tensor = image_tensor[:, slices_to_keep, :, :]
                mask_tensor = mask_tensor[:, slices_to_keep, :, :]

                # Final safety check
                if image_tensor.shape[1] != self.target_slices:
                    raise ValueError(f"After removal, slice count {image_tensor.shape[1]} not equal to target {self.target_slices}.")

        return image_tensor, mask_tensor