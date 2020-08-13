import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Preprocessing import *
from Dice_functions import *

# The LGG-Test file reads in the test data, performs the same pre-processing steps on the test set
# as the train and validate sets, and applies the test data to the chosen model. Any model can be loaded in line 28
# to be evaluated on the test data.
#
# This file required the Dice_functions and Preprocessing files.

# Reading in the unprocessed LGG data:
X_test = np.load('./Test_data/X_testLGG.npy')
Y_test = np.load('./Test_data/Y_testLGG.npy')

# Applying the three pre-processing functions:
X_test = transpose_data(X_test)
X_test, Y_test = slice_crop(X_test, Y_test)
Y_test = ground_truth_4_to_3(Y_test)

print("X_test, Y_test shape", X_test.shape, Y_test.shape)

# The following line reads in the chosen model and applies the custom loss function and metric of Dice loss
# and Dice coefficient, respectively.
model_load = tf.keras.models.load_model('model_path/model.h5',
                                        custom_objects={'dice_coefficient_loss': dice_coefficient_loss,
                                                        'dice_coefficient_function': dice_coefficient_function})


# The following section predicts on the test data and evaluates the model using the Dice score functions.

# The test_Y_pre array contains the segmentation predictions, and has been reversed from the one-hot-encoding.
test_Y_pre = np.argmax(model_load.predict(X_test), axis=-1)

# The predicted values as well as Y_test values are checked to ensure they contain all four labels:
print("test_Y_pre unique:", np.unique(test_Y_pre))
print("Y_test unique:", np.unique(Y_test))

# The prediction array and Y_test array are reshaped:
test_Y_pre = test_Y_pre.reshape(-1, 192, 192, 1)
Y_test_reshape = Y_test.reshape(-1, 192, 192, 1)

# The dice_function_loop is called to evaluate the predictions with the ground truth labels:
print("Dice scores LGG test data: ", dice_function_loop(Y_test_reshape, test_Y_pre))


# The following section calculates the confusion matrix for the test data as well as produces some sample
# segmentation images.

# The prediction and ground truth arrays are reshaped into 1D arrays to be passed to the confusion_matrix function.
test_y_pre_reshaped = test_Y_pre.reshape(-1)
test_y_test_reshaped = Y_test_reshape.reshape(-1)
cf_matrix = confusion_matrix(test_y_test_reshaped, test_y_pre_reshaped)
print(cf_matrix)

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
plt.title("LGG test data")
plt.savefig('confusion-matrix-LGG-test.png')

# The following loop takes slices 295-300 and saves their corresponding X data, Y data, and the predicted segmentation.
for i in range(295, 300):
    print('X_val ' + str(i))
    plt.imshow(X_test[i, :, :, 2])
    plt.savefig('X_test-LGG' + str(i) + '.png')
    plt.show()
    plt.imshow(test_Y_pre[i, :, :, 0])
    plt.savefig('test_Y_pre-LGG' + str(i) + '.png')
    plt.show()
    plt.imshow(Y_test_reshape[i, :, :, 0])
    plt.savefig('Y_test-LGG' + str(i) + '.png')
    plt.show()
