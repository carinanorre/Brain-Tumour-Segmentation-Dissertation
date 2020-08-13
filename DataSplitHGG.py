import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from preprocessing import *
from sklearn.model_selection import train_test_split
import os
from keras.utils import to_categorical

# DataSplitHGG uses the same functions and preprocessing as the LGG files, however re-organised in order to be able to
# apply them on the HGG data. The following code reads in the HGG data, splits the data into train/validate/test
# directly from the patients files and saves these sets. The train and validate sets are then passed through the
# process_HGG function which normalises each image and separates the MRI modality images from the ground truth images.
# The two datasets are then processed through the preprocessing functions and the ground truth data is one-hot-encoded.
# The data sets are then saved in their respective files.
#
# This code was edited and contributed to by all three group members, but adapted from the Multimodal Brain Tumour
# Segmentation GitHub repository by Aryaman Sinha. The link to the repository can be found here:
# https://github.com/as791/Multimodal-Brain-Tumor-Segmentation/blob/master/Main_file.ipynb.

# File path to HGG data:
pathHGG = "/BraTSData/MICCAI_BraTS_2018_Data_Training/HGG/"


def load_HGG(path):
    # The load_HGG function enters the directory containing the patient files, then immediately splits the data
    # into the train, validate, and test sets.
    my_dir = sorted(os.listdir(path))
    print(len(my_dir))

    X_train_HG, X_test_HG = train_test_split(my_dir, test_size=0.10, random_state=42)
    X_train_HG, X_val_HG = train_test_split(X_train_HG, test_size=0.25, random_state=42)

    return X_train_HG, X_val_HG, X_test_HG


# Calling the load_HGG function to separate the patients into train, val, test sets:
X_train_HG, X_val_HG, X_test_HG = load_HGG(pathHGG)
print("Length of train, validate, and test:", len(X_train_HG), len(X_val_HG), len(X_test_HG))


def process_HGG(path, data):
    # The process_HGG function enters the directory containing the patient files, then iterates through each patient
    # modality scan, reads in the image, organises the images by modality or ground truth, normalises each image, and
    # saves the images to a data and ground truth array.
    data_vector = []
    gt_vector = []

    for p in tqdm(data):
        data_list = sorted(os.listdir(path + p))

        # FLAIR images:
        img_itk = sitk.ReadImage(path + p + '/' + data_list[0])
        flair = sitk.GetArrayFromImage(img_itk)
        flair = normalise(flair)

        # Ground truth images:
        img_itk = sitk.ReadImage(path + p + '/' + data_list[1])
        seg = sitk.GetArrayFromImage(img_itk)

        # T1 images:
        img_itk = sitk.ReadImage(path + p + '/' + data_list[2])
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

        data_vector.append([flair, t1, t1ce, t2])
        gt_vector.append(seg)

    data_vector = np.asarray(data_vector, dtype=np.float32)
    gt_vector = np.asarray(gt_vector, dtype=np.uint8)
    return data_vector, gt_vector


# Applying the process_HGG function on the training data:
HGG_train_X, HGG_train_Y = process_HGG(pathHGG, X_train_HG)
print("Train data:", len(HGG_train_X), len(HGG_train_Y))
print("Train shapes:", HGG_train_X.shape, HGG_train_Y.shape)
print("Train types:", HGG_train_X.dtype, HGG_train_Y.dtype)

# Applying the process_HGG function on the validation data:
HGG_val_X, HGG_val_Y = process_HGG(pathHGG, X_val_HG)
print("Val data:", len(HGG_val_X), len(HGG_val_Y))
print("Val shapes:", HGG_val_X.shape, HGG_val_Y.shape)
print("Val types:", HGG_val_X.dtype, HGG_val_Y.dtype)

# Applying the processing steps from the Preprocessing file:
HGG_train_X = transpose_data(HGG_train_X)
HGG_train_X, HGG_train_Y = slice_crop(HGG_train_X, HGG_train_Y)

HGG_val_X = transpose_data(HGG_val_X)
HGG_val_X, HGG_val_Y = slice_crop(HGG_val_X, HGG_val_Y)

HGG_train_Y = ground_truth_4_to_3(HGG_train_Y)
HGG_val_Y = ground_truth_4_to_3(HGG_val_Y)

# One-hot-encoding the ground truth labels
HGG_train_Y = to_categorical(HGG_train_Y)
HGG_val_Y = to_categorical(HGG_val_Y)

print("Before saving to files data shape")
print("HGG_train_X, HGG_train_Y shape", HGG_train_X.shape, HGG_train_Y.shape)
print("HGG_val_X, HGG_val_Y shape", HGG_val_X.shape, HGG_val_Y.shape)

# Saving all the X and Y datasets separately for the train and validate sets, and saving the test data without any
# processing.
np.save('Training_data/HGG_train_X.npy', HGG_train_X)
np.save('Training_data/HGG_train_Y.npy', HGG_train_Y)
np.save('Validation_data/HGG_val_X.npy', HGG_val_X)
np.save('Validation_data/HGG_val_Y.npy', HGG_val_Y)
np.save('Test_data/HGG_test_data.npy', X_test_HG)

print("Data saved successfully")
