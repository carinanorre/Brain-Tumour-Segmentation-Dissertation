import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, BatchNormalization, concatenate, Input, Dropout,\
    Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from Dice_functions import *

# The Group-UNetModel-HGG file contains the U-Net model created as a group. The model structure, the evaluation metrics,
# and the code to create the results are all included below. This file is specifically for the HGG data.
#
# This code was edited and contributed to by all three group members, but adapted from the Multimodal Brain Tumour
# Segmentation GitHub repository by Aryaman Sinha (accessed July 1, 2020). The link to the repository can be found here:
# https://github.com/as791/Multimodal-Brain-Tumor-Segmentation/blob/master/Main_file.ipynb.

# Reading in the HGG data:
X_train = np.load('Training_data/HGG_train_X.npy')
Y_train = np.load('Training_data/HGG_train_Y.npy')

X_val = np.load('Validation_data/HGG_val_X.npy')
Y_val = np.load('Validation_data/HGG_val_Y.npy')

# U-Net model:
input_ = Input(shape=(192, 192, 4), name='input')

# Contraction path begins:
block1_conv1 = Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')(input_)
block1_conv2 = Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')(block1_conv1)
block1_norm = BatchNormalization(name='block1_batch_norm')(block1_conv2)
block1_pool = MaxPooling2D(name='block1_pool')(block1_norm)

block2_conv1 = Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')(block1_pool)
block2_conv2 = Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')(block2_conv1)
block2_norm = BatchNormalization(name='block2_batch_norm')(block2_conv2)
block2_pool = MaxPooling2D(name='block2_pool')(block2_norm)

dropout_1 = Dropout(0.2, name='encoder_dropout_1')(block2_pool)

block3_conv1 = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')(dropout_1)
block3_conv2 = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')(block3_conv1)
block3_norm = BatchNormalization(name='block3_batch_norm')(block3_conv2)
block3_pool = MaxPooling2D(name='block3_pool')(block3_norm)

block4_conv1 = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')(block3_pool)
block4_conv2 = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')(block4_conv1)
block4_norm = BatchNormalization(name='block4_batch_norm')(block4_conv2)
block4_pool = MaxPooling2D(name='block4_pool')(block4_norm)
# Contraction path ends.

# Bottom of U:
block5_conv1 = Conv2D(1024, 3, padding='same', activation='relu', name='block5_conv1')(block4_pool)

# Expansion path begins:
up_pool1 = Conv2DTranspose(1024, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool1')(block5_conv1)
merged_block1 = concatenate([block4_norm, up_pool1], name='merged_block1')
decod_block1_conv1 = Conv2D(512, 3, padding='same', activation='relu', name='decod_block1_conv1')(merged_block1)

up_pool2 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool2')(decod_block1_conv1)
merged_block2 = concatenate([block3_norm, up_pool2], name='merged_block2')
decod_block2_conv1 = Conv2D(256, 3, padding='same', activation='relu', name='decod_block2_conv1')(merged_block2)

dropout_2 = Dropout(0.2, name='decoder_dropout_1')(decod_block2_conv1)

up_pool3 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool3')(dropout_2)
merged_block3 = concatenate([block2_norm, up_pool3], name='merged_block3')
decod_block3_conv1 = Conv2D(128, 3, padding='same', activation='relu', name='decod_block3_conv1')(merged_block3)

up_pool4 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool4')(decod_block3_conv1)
merged_block4 = concatenate([block1_norm, up_pool4], name='merged_block4')
decod_block4_conv1 = Conv2D(64, 3, padding='same', activation='relu', name='decod_block4_conv1')(merged_block4)
# Expansion path ends.

# Output:
pre_output = Conv2D(64, 1, padding='same', activation='relu', name='pre_output')(decod_block4_conv1)
output = Conv2D(4, 1, padding='same', activation='softmax', name='output')(pre_output)
modelUNet = Model(inputs=input_, outputs=output)
print(modelUNet.summary())

# The model is compiled with the dice_coefficient_loss function and the dice_coefficient_function metric:
print("About to compile...")
modelUNet.compile(optimizer=Adam(lr=1e-5), loss=dice_coefficient_loss, metrics=[dice_coefficient_function])

# EarlyStopping is applied incase the model stops improving with each epoch:
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]

print("About to fit the model...")
history = modelUNet.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=5, shuffle=True,
                    callbacks=callbacks)

modelUNet.save('./Saved_models/group-HGG-UNet.h5', overwrite=True)
print("ModelSaved Successfully")
