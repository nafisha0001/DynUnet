import os
import pydicom
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, aff2axcodes, io_orientation

from utils.normalization import saveDicomAsJPEG

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