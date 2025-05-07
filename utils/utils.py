import os
import pydicom
import torch
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, aff2axcodes, io_orientation

from utils.normalization import saveDicomAsJPEG

from skimage.transform import resize

def load_nifti_as_dhw(path):
    nii = nib.load(path)
    data = nii.get_fdata()

    # Get current orientation and compute transform to RAS (standard anatomical orientation)
    orig_ornt = io_orientation(nii.affine)
    target_ornt = axcodes2ornt(('L', 'P', 'I'))  # Corresponds to [D, H, W]
    transform = ornt_transform(orig_ornt, target_ornt)

    reoriented_data = nib.orientations.apply_orientation(data, transform)

    dims = reoriented_data.shape
    sorted_axes = sorted(range(3), key=lambda x: dims[x])  # Usually depth is smallest
    if sorted_axes[0] != 0:
        reoriented_data = np.moveaxis(reoriented_data, sorted_axes[0], 0)  # Make depth first

    return reoriented_data


# def dicom_load(dicom_folder):
#     dicom_files = [
#         os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder)
#         if f.endswith('.dcm')
#     ]

#     dicom_images = [pydicom.dcmread(f) for f in dicom_files]

#     image_stack = np.stack([saveDicomAsJPEG(dcm) for dcm in dicom_images], axis=-1)
#     image_stack = np.transpose(image_stack, (2, 0, 1))   #(D,H,W)
#     print(image_stack.shape)
#     return  image_stack

def dicom_load(dicom_folder, target_shape=None):
    dicom_files = sorted([
        os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder)
        if f.endswith('.dcm')
    ])

    dicom_images = [pydicom.dcmread(f) for f in dicom_files]
    image_stack = np.stack([saveDicomAsJPEG(dcm) for dcm in dicom_images], axis=-1)  # (H, W, D)
    
    # Rearranging to (D, H, W) by default
    image_stack = np.transpose(image_stack, (2, 0, 1))  # Now (D, H, W)

    if target_shape is not None:
        if image_stack.shape != target_shape:
            # Try all permutations to find a match
            from itertools import permutations
            for perm in permutations((0, 1, 2)):
                permuted = np.transpose(image_stack, perm)
                if permuted.shape == target_shape:
                    print(f"[INFO] Transposing image_stack from {image_stack.shape} to {permuted.shape} using order {perm}")
                    image_stack = permuted
                    break
    #         else:
    #             raise ValueError(f"Unable to match image_stack shape {image_stack.shape} to target shape {target_shape}")
    #     else:
    #         print("[INFO] image_stack shape already matches target shape.")
    
    # print(f"[INFO] Final image_stack shape: {image_stack.shape}")
    return image_stack


# def nifti_folder_load(nifti_path):

#     # Load as (D, H, W)
#     volume = load_nifti_as_dhw(nifti_path)
#     processed_volume = saveDicomAsJPEG(volume)

    # Apply saveDicomAsJPEG to each slice
    # processed_slices = []
    # for i in range(volume.shape[0]):
    #     slice_2d = volume[i, :, :]
    #     processed_slice = saveDicomAsJPEG(slice_2d)
    #     processed_slices.append(processed_slice)

    # # Stack along depth axis to get (D, H, W)
    # processed_volume = np.stack(processed_slices, axis=0)

    # return processed_volume



def resample_segmentation_to_image(segmentation, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    return resampler.Execute(segmentation)


def nearest_multiple_of_32(n):
    return ((n + 31) // 32) * 32

# def custom_collate(batch):
#     max_slices = max(x.shape[1] for x, y in batch) 
#     max_depth = nearest_multiple_of_32(max_slices) 
    
#     newImageVolume = []
#     newMaskVolume = []

#     for image_tensor, mask_tensor in batch:
#         current_slices = image_tensor.shape[1]

#         if current_slices < max_depth:
#             # Duplicate labeled slices
#             labeled_slices = []
#             for i in range(current_slices):
#                 labeled_slices.append(i)

#             while current_slices < max_depth:
#                 for i in labeled_slices:
#                     if current_slices < max_depth:
#                         image_tensor = torch.cat((image_tensor, image_tensor[:, i:i+1, :, :]), dim=1)
#                         mask_tensor = torch.cat((mask_tensor, mask_tensor[:, i:i+1, :, :]), dim=1)
#                         current_slices += 1

#         # Append to batch
#         newImageVolume.append(image_tensor)
#         newMaskVolume.append(mask_tensor)

#     # Stack batch tensors
#     newImageVolume = torch.stack(newImageVolume, dim=0)
#     newMaskVolume = torch.stack(newMaskVolume, dim=0)

#     return newImageVolume, newMaskVolume



# blank Slices:
def custom_collate(batch):
    max_slices = max(x.shape[1] for x, y in batch) 
    max_depth = nearest_multiple_of_32(max_slices) 

    newImageVolume = []
    newMaskVolume = []

    for image_tensor, mask_tensor in batch:
        current_slices = image_tensor.shape[1]

        if current_slices < max_depth:
            # Calculate number of blank slices needed
            pad_slices = max_depth - current_slices

            # Create blank image and mask slices
            blank_image = torch.zeros((image_tensor.shape[0], pad_slices, image_tensor.shape[2], image_tensor.shape[3]), dtype=image_tensor.dtype)
            blank_mask = torch.zeros((mask_tensor.shape[0], pad_slices, mask_tensor.shape[2], mask_tensor.shape[3]), dtype=mask_tensor.dtype)

            # Pad with blank slices
            image_tensor = torch.cat((image_tensor, blank_image), dim=1)
            mask_tensor = torch.cat((mask_tensor, blank_mask), dim=1)

        # Append to batch
        newImageVolume.append(image_tensor)
        newMaskVolume.append(mask_tensor)

    # Stack batch tensors
    newImageVolume = torch.stack(newImageVolume, dim=0)
    newMaskVolume = torch.stack(newMaskVolume, dim=0)

    return newImageVolume, newMaskVolume





#  Resampling code


target_spacing= [1.0, 1.0, 1.0]  # Target spacing in mm (x, y, z)

def load_spacing(image_obj):
    spacing= image_obj.GetSpacing()
    spacing= list(spacing)
    return [spacing[2], spacing[1], spacing[0]]

def resample_pair(image, label, spacing):
    shape = calculate_new_shape(spacing, image.shape)
    if check_anisotrophy(spacing):
        image = resample_anisotrophic_image(image, shape)
        if label is not None:
            label = resample_anisotrophic_label(label, shape)
    else:
        image = resample_regular_image(image, shape)
        if label is not None:
            label = resample_regular_label(label, shape)
    image = image.astype(np.float32)
    if label is not None:
        label = label.astype(np.uint8)
    return image, label

def calculate_new_shape(spacing, shape):
    spacing_ratio = np.array(spacing) / np.array(target_spacing)
    new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
    return new_shape

def check_anisotrophy(spacing):
    def check(spacing):
        return np.max(spacing) / np.min(spacing) >= 3

    return check(spacing) or check(target_spacing)

def resample_anisotrophic_image(image, shape):
    resized = [resize_fn(i, shape[1:], 3, "edge") for i in image]
    resized = np.stack(resized, axis=0)
    resized = resize_fn(resized, shape, 0, "constant")
    return resized

def resample_anisotrophic_label(label, shape):
    depth = label.shape[0]
    reshaped = np.zeros(shape, dtype=np.uint8)
    shape_2d = shape[1:]
    reshaped_2d = np.zeros((depth, *shape_2d), dtype=np.uint8)
    n_class = np.max(label)
    for class_ in range(1, n_class + 1):
        for depth_ in range(depth):
            mask = label[depth_] == class_
            resized_2d = resize_fn(mask.astype(float), shape_2d, 1, "edge")
            reshaped_2d[depth_][resized_2d >= 0.5] = class_

    for class_ in range(1, n_class + 1):
        mask = reshaped_2d == class_
        resized = resize_fn(mask.astype(float), shape, 0, "constant")
        reshaped[resized >= 0.5] = class_
    return reshaped

def resample_regular_image(image, shape):
    resized = (resize_fn(image, shape, 3, "edge"))
    return resized

def resample_regular_label(label, shape):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    for class_ in range(1, n_class + 1):
        mask = label == class_
        resized = resize_fn(mask.astype(float), shape, 1, "edge")
        reshaped[resized >= 0.5] = class_
    return reshaped

def resize_fn(image, shape, order, mode):
    return resize(image, shape, order=order, mode=mode, cval=0, clip=True, anti_aliasing=False)