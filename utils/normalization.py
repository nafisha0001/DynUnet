import tempfile
import cv2 as cv2
import numpy as np
import pydicom as dicom
import os

def window_image(img, window_center, window_width, rescale=True):
    # img = (img * slope + intercept)  # for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    if rescale:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img

def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)
 
def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    # data[('0028', '1052')].value,  # intercept
                    # data[('0028', '1053')].value]  # slope
                   ]
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
 
def getData(dicom_object, attribute):
    data_elem = getattr(dicom_object, attribute, None)
    return data_elem
 

def getDicomFIleIds(dicom_object_path):
    ds = dicom.dcmread(dicom_object_path)
    study_instance_uid = getData(ds, 'StudyInstanceUID')
    series_instance_uid = getData(ds, 'SeriesInstanceUID')
    sop_instance_uid = getData(ds, 'SOPInstanceUID')
    return sop_instance_uid, study_instance_uid, series_instance_uid


def saveDicomAsJPEG(ds):
    # ds = dicom.dcmread(image_path)                   
    window_center, window_width= get_windowing(ds)
    output = window_image(ds.pixel_array.astype(float), window_center, window_width, rescale=False)
    new_image = output.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    return scaled_image