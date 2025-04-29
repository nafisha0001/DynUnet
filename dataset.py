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

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.data_dir, self.mask_filenames[idx])
        print(f"Reading image: {image_path}")
        print(f"Reading mask: {mask_path}")

        mask = load_nifti_as_dhw(mask_path)
        image = dicom_load(image_path, mask.shape)
        print(image.shape, mask.shape)

        slices, _, _ = image.shape
        image_tensor = []
        mask_tensor = []

        for i in range(slices):
            slice_img = image[i, :, :].astype(np.uint8)
            slice_mask = mask[i, :, :].astype(np.uint8)

            if self.transform:
                transformed = self.transform(image=slice_img, mask=slice_mask)
                slice_img = transformed["image"]
                slice_mask = transformed["mask"]
            else:
                slice_img = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0)
                slice_mask = torch.tensor(slice_mask, dtype=torch.float32).unsqueeze(0)

            image_tensor.append(slice_img)
            mask_tensor.append(slice_mask)

        image_tensor = torch.stack(image_tensor)  # (D, C, H, W)
        mask_tensor = torch.stack(mask_tensor)    # (D, C, H, W)

        # Change to (C, D, H, W)
        image_tensor = image_tensor.permute(1, 0, 2, 3)  # (C, D, H, W)
        mask_tensor = mask_tensor.unsqueeze(0)    # (C, D, H, W)

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