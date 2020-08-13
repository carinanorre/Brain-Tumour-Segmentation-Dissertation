import numpy as np
from Dice_functions import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# The HGG-UNetModel-Eval file loads any HGG model and evaluates the performance on the train and validate sets
# using the Dice score functions, confusion matrices, and prediction images.

# Reading in the HGG data:
X_train = np.load('Training_data/HGG_train_X.npy')
Y_train = np.load('Training_data/HGG_train_Y.npy')

X_val = np.load('Validation_data/HGG_val_X.npy')
Y_val = np.load('Validation_data/HGG_val_Y.npy')

# Any HGG model can be read in while applying the custom functions:
modelUNet = tf.keras.models.load_model('./Saved_models/individual-model-HGG.h5',
                                            custom_objects={'dice_coefficient_loss': dice_coefficient_loss,
                                                            'dice_coefficient_function': dice_coefficient_function})


# In the next section, the model is applied to the validation data and train data to evaluate its performance.

# VALIDATION DATA:
# Reversing the one-hot-encoding for both the prediction array and ground truth array:
val_Y_pre = np.argmax(modelUNet.predict(X_val), axis=-1)
Y_val = np.argmax(Y_val, axis=-1)

# The predicted values as well as Y_val values are checked to ensure they contain all four labels:
print("val_Y_pre unique:", np.unique(val_Y_pre))
print("Y_val unique:", np.unique(Y_val))

# The prediction array and Y_val array are reshaped:
val_Y_pre = val_Y_pre.reshape(-1, 192, 192, 1)
Y_val_reshape = Y_val.reshape(-1, 192, 192, 1)

# The dice_function_loop is called to evaluate the predictions with the ground truth labels:
print("Dice scores HGG validation data: ", dice_function_loop(Y_val_reshape, val_Y_pre))

# The prediction and ground truth arrays are reshaped into 1D arrays to be passed to the confusion_matrix function.
val_y_pre_reshaped = val_Y_pre.reshape(-1)
val_y_val_reshaped = Y_val_reshape.reshape(-1)
cf_matrix = confusion_matrix(val_y_val_reshaped, val_y_pre_reshaped)
print(confusion_matrix(val_y_val_reshaped, val_y_pre_reshaped))

# The following loop iterates through the confusion matrix and calculates the percentage of pixels segmented into
# each class.
print("Class percent confusion matrix:")
cf_matrix_classpercent = np.zeros(shape=(4, 4))
for j in range(4):
    for i in range(4):
        cf_matrix_classpercent[j][i] = cf_matrix[j][i] / (cf_matrix[j][0]+cf_matrix[j][1]+cf_matrix[j][2]+cf_matrix[j][3])
print(cf_matrix_classpercent)

# The confusion matrix is turned into a heatmap and saved to the current directory.
sns.heatmap(cf_matrix_classpercent, annot=True, fmt='.2%', cmap='Blues', cbar=False)
plt.xlabel("Predicted values")
plt.ylabel("Ground truth values")
plt.title("HGG validation data")
plt.savefig('confusion-matrix-HGG-val.png')

# The following loop takes slices 295-300 and saves their corresponding X data, Y data, and the predicted segmentation.
for i in range(295, 300):  # 470,475
    print('X_val ' + str(i))
    plt.imshow(X_val[i, :, :, 2])
    plt.savefig('X_val ' + str(i))
    plt.show()
    plt.imshow(val_Y_pre[i, :, :, 0])
    plt.savefig('val_Y_pre ' + str(i))
    plt.show()
    plt.imshow(Y_val_reshape[i, :, :, 0])
    plt.savefig('Y_val ' + str(i))
    plt.show()


# TRAIN DATA:
# Reversing the one-hot-encoding for both the prediction array and ground truth array:
train_Y_pre = np.argmax(modelUNet.predict(X_train), axis=-1)
Y_train = np.argmax(Y_train, axis=-1)

# The predicted values as well as Y_val values are checked to ensure they contain all four labels:
print("train_Y_pre unique:", np.unique(train_Y_pre))
print("Y_train unique:", np.unique(Y_train))

# The prediction array and Y_val array are reshaped:
train_Y_pre = train_Y_pre.reshape(-1, 192, 192, 1)
Y_train_reshape = Y_train.reshape(-1, 192, 192, 1)  # should not have to do this

# The dice_function_loop is called to evaluate the predictions with the ground truth labels:
print("Dice scores HGG train data: ", dice_function_loop(Y_train_reshape, train_Y_pre))

# The prediction and ground truth arrays are reshaped into 1D arrays to be passed to the confusion_matrix function.
train_y_pre_reshaped = train_Y_pre.reshape(-1)
train_y_train_reshaped = Y_train_reshape.reshape(-1)
cf_matrix_train = confusion_matrix(train_y_train_reshaped, train_y_pre_reshaped)
print(cf_matrix_train)

# The following loop iterates through the confusion matrix and calculates the percentage of pixels segmented into
# each class.
print("Class percent confusion matrix:")
cf_matrix_classpercent = np.zeros(shape=(4, 4))
for j in range(4):
    for i in range(4):
        cf_matrix_classpercent[j][i] = cf_matrix_train[j][i] / (cf_matrix_train[j][0]+cf_matrix_train[j][1] +
                                                                cf_matrix_train[j][2]+cf_matrix_train[j][3])
print(cf_matrix_classpercent)

# The confusion matrix is turned into a heatmap and saved to the current directory.
sns.heatmap(cf_matrix_classpercent, annot=True, fmt='.2%', cmap='Blues', cbar=False)
plt.xlabel("Predicted Values")
plt.ylabel("Ground truth values")
plt.title("HGG train data")
plt.savefig('confusion-matrix-HGG-train.png')


# The following loop takes slices 295-300 and saves their corresponding X data, Y data, and the predicted segmentation.
for i in range(295, 300):
    print('X_train-HGG' + str(i))
    plt.imshow(X_train[i, :, :, 2])
    plt.savefig('X_train-HGG' + str(i))
    plt.show()
    plt.imshow(train_Y_pre[i, :, :, 0])
    plt.savefig('train_Y_pre-HGG' + str(i))
    plt.show()
    plt.imshow(Y_train_reshape[i, :, :, 0])
    plt.savefig('Y_train-HGG' + str(i))
    plt.show()
