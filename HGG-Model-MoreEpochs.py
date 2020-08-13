import numpy as np
import tensorflow as tf
from Dice_functions import *

# The HGG-Model-MoreEpochs file reads in the chosen HGG model and continues training it for 5 more epochs with the
# same loss function, dice metric, and data. This method was chosen to train the HGG data as a work around for the
# problems caused by the large quantity of HGG data. The model is then saved again after being trained for 5 epochs,
# and can be re-loaded to continue training. The final saved model can then be read into the HGG-UNetModel-Eval file
# to evaluate it on the train and validate sets, and read into the HGG-Test file to evaluate it on the test set.

# Reading in the HGG data:
X_train = np.load('Training_data/HGG_train_X.npy')
Y_train = np.load('Training_data/HGG_train_Y.npy')

X_val = np.load('Validation_data/HGG_val_X.npy')
Y_val = np.load('Validation_data/HGG_val_Y.npy')

# Loading the saved HGG model:
model_more_epochs = tf.keras.models.load_model('HGG-model.h5',
                                               custom_objects={'dice_coefficient_loss': dice_coefficient_loss,
                                                                'dice_coefficient_function': dice_coefficient_function})

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]
history = model_more_epochs.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=5,
                     shuffle=True, callbacks=callbacks)

# Saving the new model after an additional training of 5 epochs:
model_more_epochs.save('HGG-model-more-epochs.h5', overwrite=True)
print("ModelSaved Successfully")
