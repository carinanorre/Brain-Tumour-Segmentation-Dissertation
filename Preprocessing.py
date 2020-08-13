import numpy as np

# The Preprocessing file contains the functions called in the DataLoad and DataSplit files. These functions include the
# normalise function, the transpose_data function, the slice_crop function, and the ground_truth_4_to_3 function.
#
# This code was edited and contributed to by all three group members, but adapted from the Multimodal Brain Tumour
# Segmentation GitHub repository by Aryaman Sinha. The link to the repository can be found here:
# https://github.com/as791/Multimodal-Brain-Tumor-Segmentation/blob/master/Main_file.ipynb.


def normalise(img, bottom=99, top=1):
    # The normalise function subtracts the mean intensity from the image and divides by the standard deviation.
    b = np.percentile(img, bottom)

    t = np.percentile(img, top)

    clipImage = np.clip(img, t, b)

    if np.std(clipImage) == 0:
        return clipImage
    else:
        normImage = (clipImage - np.mean(clipImage)) / np.std(clipImage)
        return normImage


def transpose_data(data):
    # The transposeData function reorders the dimensions of the array.
    data = np.transpose(data, (0, 2, 3, 4, 1))
    return data


def slice_crop(data, gt):
    # The sliceCrop function firstly extracts only slices 30-120 of the 3D image, then reshapes the image from 255x255
    # to 192x192.
    data = data[:, 30:120, 30:222, 30:222, :].reshape([-1, 192, 192, 4])
    gt = gt[:, 30:120, 30:222, 30:222].reshape([-1, 192, 192, 1])
    return data, gt


def ground_truth_4_to_3(gt):
    # The groundTruth4to3 function relabels the ground truth class 4 pixels as class 3. Class 4 represents the
    # enhancing tumour region, which is then changed to class 3. This change is done to enable the one-hot-encoding
    # of the ground truth labels.
    gt[np.where(gt == 4)] = 3
    return gt
