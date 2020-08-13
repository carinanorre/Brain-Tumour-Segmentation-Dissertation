import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, BatchNormalization, concatenate, Input, \
    Activation, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from Dice_functions import *

# The Individual-UNetModel-LGG file contains the U-Net model explored individually for this project. This file evaluates
# the LGG dataset. The U-Net model below contains all the additional 2D convolutional layers added onto the base,
# but they are clearly identified as optional additional.
#
# This model is based on the original Ronneberger et al. U-Net design [1], and the code is adapted from the
# Multimodal Brain Tumour Segmentation GitHub repository by Aryaman Sinha [2] (accessed July 1, 2020) as well as from
# the UNET-TGS GitHub repository by Harshall Lamba [3] (accessed July 20, 2020).
#
# [1] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation.
# CoRR. 2015;abs/1505.04597.  Available from:http://arxiv.org/abs/1505.04597.
# [2] https://github.com/as791/Multimodal-Brain-Tumor-Segmentation/blob/master/Main_file.ipynb
# [3] https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb

# Reading in the train and validate datasets:
X_train = np.load('./Training_data/X_trainLGG.npy')
Y_train = np.load('./Training_data/Y_trainLGG.npy')

X_val = np.load('./Validation_data/X_valLGG.npy')
Y_val = np.load('./Validation_data/Y_valLGG.npy')


def conv2d_pair(input, n_filters, kernel_size=3):
    # The conv2d_pair function creates a pair of 2D convolutional layers with n_filters as the number of filters,
    # and kernel_size as the desired filter size.
    conv2D_layer = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input)
    batch_norm_layer = BatchNormalization()(conv2D_layer)
    activation_layer = Activation('relu')(batch_norm_layer)

    conv2D_layer2 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(activation_layer)
    batch_norm_layer2 = BatchNormalization()(conv2D_layer2)
    final_layer = Activation('relu')(batch_norm_layer2)

    return final_layer


def individual_unet(input_img, n_filters=16):
    # The individual_unet function creates the U-Net model with a starting filter number of 16. The pairs of 2D
    # convolutional layers that are added on to the base structure are noted below.

    # Contraction path begins:
    block1_c1 = conv2d_pair(input_img, n_filters * 1, kernel_size=3)
    block1_c2 = conv2d_pair(block1_c1, n_filters * 1, kernel_size=3)  # Optional additional
    block1_maxpool = MaxPooling2D((2, 2))(block1_c2)

    block2_c1 = conv2d_pair(block1_maxpool, n_filters * 2, kernel_size=3)
    block2_c2 = conv2d_pair(block2_c1, n_filters * 2, kernel_size=3)  # Optional additional
    block2_maxpool = MaxPooling2D((2, 2))(block2_c2)

    block3_c1 = conv2d_pair(block2_maxpool, n_filters * 4, kernel_size=3)
    block3_c2 = conv2d_pair(block3_c1, n_filters * 4, kernel_size=3)  # Optional additional
    block3_maxpool = MaxPooling2D((2, 2))(block3_c2)

    block4_c1 = conv2d_pair(block3_maxpool, n_filters * 8, kernel_size=3)
    block4_c2 = conv2d_pair(block4_c1, n_filters * 8, kernel_size=3)  # Optional additional
    block4_maxpool = MaxPooling2D((2, 2))(block4_c2)
    # Contraction path ends.

    block5_c1 = conv2d_pair(block4_maxpool, n_filters=n_filters * 16, kernel_size=3)
    block5_c2 = conv2d_pair(block5_c1, n_filters=n_filters * 16, kernel_size=3)  # Optional additional

    # Expansion path begins:
    block6_up = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(block5_c2)
    block6_concat = concatenate([block6_up, block4_c2])
    block6_c1 = conv2d_pair(block6_concat, n_filters * 8, kernel_size=3)
    block6_c2 = conv2d_pair(block6_c1, n_filters * 8, kernel_size=3)  # Optional additional

    block7_up = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(block6_c2)
    block7_concat = concatenate([block7_up, block3_c2])
    block7_c1 = conv2d_pair(block7_concat, n_filters * 4, kernel_size=3)
    block7_c2 = conv2d_pair(block7_c1, n_filters * 4, kernel_size=3)  # Optional additional

    block8_up = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(block7_c2)
    block8_concat = concatenate([block8_up, block2_c2])
    block8_c1 = conv2d_pair(block8_concat, n_filters * 2, kernel_size=3)
    block8_c2 = conv2d_pair(block8_c1, n_filters * 2, kernel_size=3)  # Optional additional

    block9_up = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(block8_c2)
    block9_concat = concatenate([block9_up, block1_c1])
    block9_c1 = conv2d_pair(block9_concat, n_filters / 4, kernel_size=3)
    block9_c2 = conv2d_pair(block9_c1, n_filters / 4, kernel_size=3)  # Optional additional
    # Expansion path ends.

    outputs = Conv2D(4, (1, 1), activation='softmax')(block9_c2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# The individual_unet model is called with the corresponding input size of the image data:
modelUNet = individual_unet(Input(shape=(192, 192, 4)))
print(modelUNet.summary())

# The model is compiled with the dice_coefficient_loss function and the dice_coefficient_function metric:
print("About to compile...")
modelUNet.compile(optimizer=Adam(lr=1e-5), loss=dice_coefficient_loss, metrics=[dice_coefficient_function])

print("About to fit the model...")
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]
history = modelUNet.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=40, shuffle=True,
                        callbacks=callbacks)

modelUNet.save('./Saved_models/individual-model-LGG.h5', overwrite=True)
print("ModelSaved Successfully")

