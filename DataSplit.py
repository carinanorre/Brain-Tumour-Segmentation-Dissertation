import numpy as np
from Preprocessing import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# The DataSplit file reads in the data produced from DataLoad and conducts the train/validate/test split. This code only
# works on the LGG data. After creating the split, each function from Preprocessing is applied to each group of data.
# The ground truth is one-hot-encoded to reflected the non-ordered data. The preprocessed data is then saved into their
# respective files.
#
# This code was edited and contributed to by all three group members, but adapted from the Multimodal Brain Tumour
# Segmentation GitHub repository by Aryaman Sinha. The link to the repository can be found here:
# https://github.com/as791/Multimodal-Brain-Tumor-Segmentation/blob/master/Main_file.ipynb.

# Read in data:
data = np.load('LG_data.npy')
gt = np.load('LG_gt.npy')

# The percentage split for the train, validate, and test sets is 70, 20, 10.
X_train, X_test, Y_train, Y_test = train_test_split(data, gt, test_size=0.10, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

print("After split X_train, Y_train shape", X_train.shape, Y_train.shape)
print("After split X_val, Y_val shape", X_val.shape, Y_val.shape)
print("After split X_test, Y_test shape", X_test.shape, Y_test.shape)

# After split: X_train, Y_train shape (50, 4, 155, 240, 240) (50, 155, 240, 240).
# After split: X_val, Y_val shape (17, 4, 155, 240, 240) (17, 155, 240, 240).
# After split: X_test, Y_test shape (8, 4, 155, 240, 240) (8, 155, 240, 240).
# This indicates that 50 patients have been allocated to the training set, 17 to the validation set
# and 8 to the test set.

# Preprocessing only train and validate sets:
X_train = transpose_data(X_train)
X_train, Y_train = slice_crop(X_train, Y_train)

X_val = transpose_data(X_val)
X_val, Y_val = slice_crop(X_val, Y_val)

Y_train = ground_truth_4_to_3(Y_train)
Y_val = ground_truth_4_to_3(Y_val)

# One-hot-encoding the ground truth labels for the train and validate sets:
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)

# Checking the shapes of the train and validate sets:
print("X_train, Y_train shape", X_train.shape, Y_train.shape)
print("X_val, Y_val shape", X_val.shape, Y_val.shape)

# Saving each data set:
np.save('./Training_data/X_trainLGG.npy', X_train)
np.save('./Training_data/Y_trainLGG.npy', Y_train)
np.save('./Validation_data/X_valLGG.npy', X_val)
np.save('./Validation_data/Y_valLGG.npy', Y_val)
np.save('./Test_data/X_testLGG.npy', X_test)
np.save('./Test_data/Y_testLGG.npy', Y_test)

print("Data saved successfully")
