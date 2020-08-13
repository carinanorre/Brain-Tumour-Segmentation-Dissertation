import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Dice_functions import *
from DataSplitHGG import process_HGG
from Preprocessing import *
import seaborn as sns

# The HGG-Test file reads in the raw test data and applied on the processing steps that were applied to the train and
# validate sets. The chosen HGG model is then read in and evaluated on the test data using Dice scores, confusion matrix
# and prediction images.
#
# This file required the Dice_functions, DataSplitHGG, and Preprocessing files.

# Reading in the HGG data:
HGG_data = np.load('Test_data/HGG_test_data.npy')

# File path:
pathHGG = "/BraTSData/MICCAI_BraTS_2018_Data_Training/HGG/"

# Applying the process_HGG function from the DataSplitHGG file on the test data:
HGG_test_X, HGG_test_Y = process_HGG(pathHGG, HGG_data)
print("Test data:", len(HGG_test_X), len(HGG_test_Y))
print("Test shapes:", HGG_test_X.shape, HGG_test_Y.shape)
print("Test types:", HGG_test_X.dtype, HGG_test_Y.dtype)

# Applying the preprocessing steps on the test data:
HGG_test_X = transpose_data(HGG_test_X)
HGG_test_X, HGG_test_Y = slice_crop(HGG_test_X, HGG_test_Y)
HGG_test_Y = ground_truth_4_to_3(HGG_test_Y)

# Loading in the HGG model:
modelUNetHGG = tf.keras.models.load_model('./Saved_models/individual-model-HGG.h5',
                                               custom_objects={'dice_coefficient_loss': dice_coefficient_loss,
                                                               'dice_coefficient_function': dice_coefficient_function})

# The following section predicts on the test data and evaluates the model using the Dice score functions.

# The test_Y_pre array contains the segmentation predictions, and has been reversed from the one-hot-encoding.
test_Y_pre = np.argmax(modelUNetHGG.predict(HGG_test_X), axis=-1)

# The predicted values as well as Y_test values are checked to ensure they contain all four labels:
print("test_Y_pre unique:", np.unique(test_Y_pre))
print("Y_test unique:", np.unique(HGG_test_Y))

# The prediction array and Y_test array are reshaped:
test_Y_pre = test_Y_pre.reshape(-1, 192, 192, 1)
Y_test_reshape = HGG_test_Y.reshape(-1, 192, 192, 1)


# The dice_function_loop is called to evaluate the predictions with the ground truth labels:
print("Dice scores HGG test data: ", dice_function_loop(Y_test_reshape, test_Y_pre))

# The following section calculates the confusion matrix for the test data as well as produces some sample
# segmentation images.

# The prediction and ground truth arrays are reshaped into 1D arrays to be passed to the confusion_matrix function.
test_y_pre_reshaped = test_Y_pre.reshape(-1)
test_y_val_reshaped = Y_test_reshape.reshape(-1)
cf_matrix = confusion_matrix(test_y_val_reshaped, test_y_pre_reshaped)
print(confusion_matrix(test_y_val_reshaped, test_y_pre_reshaped))

# The following loop iterates through the confusion matrix and calculates the percentage of pixels segmented into
# each class.
print("Class percent confusion matrix:")
cf_matrix_classpercent = np.zeros(shape=(4, 4))
for j in range(4):
    for i in range(4):
        cf_matrix_classpercent[j][i] = cf_matrix[j][i] / (cf_matrix[j][0]+cf_matrix[j][1]+cf_matrix[j][2]+cf_matrix[j][3])
        cf_matrix_classpercent[j][i] = '{:.3f}'.format(cf_matrix_classpercent[j][i])
print(cf_matrix_classpercent)

# The confusion matrix is turned into a heatmap and saved to the current directory.
sns.heatmap(cf_matrix_classpercent, annot=True, fmt='.2%', cmap='Blues', cbar=False)
plt.xlabel("Predicted values")
plt.ylabel("Ground truth values")
plt.title("HGG test data")
plt.savefig('confusion-matrix-HGG-test.png')

# The following loop takes slices 295-300 and saves their corresponding X data, Y data, and the predicted segmentation.
for i in range(295, 300):
    print('X_val ' + str(i))
    plt.imshow(HGG_test_X[i, :, :, 2])
    plt.savefig('X_test-HGG' + str(i) + '.png')
    plt.show()
    plt.imshow(test_Y_pre[i, :, :, 0])
    plt.savefig('test_Y_pre-HGG' + str(i) + '.png')
    plt.show()
    plt.imshow(Y_test_reshape[i, :, :, 0])
    plt.savefig('Y-test-HGG' + str(i) + '.png')
    plt.show()
