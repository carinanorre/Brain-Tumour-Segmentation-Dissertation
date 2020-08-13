import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from Preprocessing import normalise
import os

# The DataLoad file reads in the data from the defined path and iterates through each patient's file using the load_data
# function to collect and organise the MRI modality scans and the ground truth annotation scans into separate folders.
# In the process of reading in the files, the load_data function also normalises the images. These files are then saved
# to the current directory to be read in by following python files.
#
# This DataLoad file requires the functions defined in the Preprocessing file.
# This code only works on the LGG data. This code was edited and contributed to by all three group members, but adapted
# from the Multimodal Brain Tumour Segmentation GitHub repository by Aryaman Sinha. The link to the repository can be
# found here: https://github.com/as791/Multimodal-Brain-Tumor-Segmentation/blob/master/Main_file.ipynb.

# File path:
pathLGG = "../BraTSData/MICCAI_BraTS_2018_Data_Training/LGG/"


def load_data(path):
    # The load_data function enters the directory containing the patient files, then iterates through each patient
    # modality scan, reads in the image, organises the images by modality or ground truth, normalises each image, and
    # saves the images to a data and ground truth array.
    my_dir = sorted(os.listdir(path))

    data = []
    gt = []

    for p in tqdm(my_dir):
        data_list = sorted(os.listdir(path+p))

        # FLAIR images:
        img_itk = sitk.ReadImage(path + p + '/' + data_list[0])
        flair = sitk.GetArrayFromImage(img_itk)
        flair = normalise(flair)

        # Ground truth images:
        img_itk = sitk.ReadImage(path + p + '/' + data_list[1])
        seg = sitk.GetArrayFromImage(img_itk)

        # T1 images:
        img_itk = sitk.ReadImage(path + p + '/'+ data_list[2])
        t1 = sitk.GetArrayFromImage(img_itk)
        t1 = normalise(t1)

        # T1ce (T1Gd) images:
        img_itk = sitk.ReadImage(path + p + '/' + data_list[3])
        t1ce = sitk.GetArrayFromImage(img_itk)
        t1ce = normalise(t1ce)

        # T2 images:
        img_itk = sitk.ReadImage(path + p + '/' + data_list[4])
        t2 = sitk.GetArrayFromImage(img_itk)
        t2 = normalise(t2)

        data.append([flair, t1, t1ce, t2])
        gt.append(seg)

    data = np.asarray(data, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.uint8)
    return data, gt


# Calling the load_data function using the path to the raw data, and saving the loaded images
# into data and ground truth arrays.
dataLGG, gtLGG = load_data(pathLGG)

# Checking the shapes and data types of the saved arrays:
print("dataLGG.shape", dataLGG.shape)
print("gtLGG.shape", gtLGG.shape)
print("dataLGG.dtype", dataLGG.dtype)
print("gtLGG.dtype", gtLGG.dtype)

# Saving the final data sets to the current directory:
np.save('LG_data.npy', dataLGG)
np.save('LG_gt.npy', gtLGG)
