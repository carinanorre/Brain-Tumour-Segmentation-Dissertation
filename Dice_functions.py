import numpy as np
from tensorflow.keras import backend as K

# This Dice_functions file contains all the Dice score calculations needed for the different models. Except where
# otherwise stated, the functions below were adapted from the brats_2019 GitHub repository, which can be found at:
# https://github.com/woodywff/brats_2019/blob/master/demo_task1/evaluate.py, and was accessed from July 3, 2020.


# The dice_coefficient_function code was adapted from a discussion on Github about generalised dice loss for multi-class
# segmentation, which can be found at: https://github.com/keras-team/keras/issues/9395 (gattia).
def dice_coefficient_function(y_true, y_pred, smooth=1e-7):
    # The dice_coefficient_function is the main function used to compile the models. It returns the Dice score
    # for all the labels.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    # The dice_coefficient_loss function returns the loss function used to compile the models.
    return 1-dice_coefficient_function(y_true, y_pred)


# The following four functions isolate the different tumour regions through the labelled classes.


def get_whole_tumour(data):
    # The get_whole_tumour function isolates the pixels classed as 1,2,3, to create the whole tumour.
    return data > 0


def get_tumour_core(data):
    # The get_tumour_core function isolates the pixels classed as 1 and 3 to create the core tumour.
    return np.logical_or(data == 1, data == 3)


def get_enhancing_tumour(data):
    # The get_enhancing_tumour function isolates the pixels classed as 3 to create the enhancing tumour.
    return data == 3


def get_background(data):
    # The get_background function returns all the background pixels.
    return data == 0


def dice_coefficient_tumour_regions(truth, prediction):
    # This dice_coefficient_tumour_regions functions calculates the dice score for the arrays that are no longer
    # one-hot-encoded.
    if np.sum(truth) + np.sum(prediction) == 0:
        return 1
    else:
        return 2 * np.sum(truth * prediction) / (np.sum(truth) + np.sum(prediction))


# The dice_function_loop function was written by the group members of this project.
def dice_function_loop(ground_truth, preds):
    # This dice_function_loop prints each region's dice score using the above functions.
    tumour_part = ("Whole tumour", "Tumour core", "Enhancing tumour", "Background")
    tumour_part_functions = (get_whole_tumour, get_tumour_core, get_enhancing_tumour, get_background)
    rows = list()
    print(tumour_part)
    rows.append([dice_coefficient_tumour_regions(func(ground_truth), func(preds)) for func in tumour_part_functions])
    print(rows)
