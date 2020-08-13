# Brain-Tumour-Segmentation-Dissertation

The goal of the Brain Tumour Segmentation (in MRI) project is to explore the state-of-the-art methods in brain tumour segmentation and apply a chosen model to the BraTS 2018 Challenge data, from the Perelman School of Medicine, University of Pennsylvania. The data contains 210 high-grade gliomas (HGG) patients and 75 low-grade gliomas (LGG) patients. This project was partially completed as a group, and partially completed individually. The group members include Neha Ghatia, Daria Tkachova, and Carina Norregaard. The chosen model to explore was the U-Net model, originally developed by Ronneberger et al. The group part of the project consisted of pre-processing the data and building a base U-Net to segment the tumours. The individual part of the project consisted of adapting Ronneberger et al.'s model by adding 2D convolutional layers to deepen the model and potentially extract more features from the data. All the models built in this project were evaluated using Dice scores, confusion matrices, and prediction segmentation images.

The final project consists of a total of 14 python files. The files can be split up according to the type of tumour data being applied, LGG or HGG.

## LGG data
To initially read in the LGG data and normalise the data, file DataLoad.py is used, where the relative path to the data is given. The file Preprocessing.py is required as an import into this file since it contains the pre-processing functions such as normalise, transpose_data, slice_crop, and ground_truth_4_to_3. DataLoad.py saves the MRI data as arrays, which can then be used as input into the DataSplit.py file. In the DataSplit.py file, the data is split into train, validate, and test sets by patient. The transpose_data, slice_crop, and ground_truth_4_to_3 functions are applied on the data to pre-process it further.

#### Group model
After the LGG data has been read in and pre-processed, it can be applied to the group based U-Net model, contained in file Group-UNetModel-LGG.py. This U-Net model file contains the architecture of the model, as well as compiles and fits it to the LGG training and validation data. The model is compiled with a Dice coefficient loss function as well as Dice coefficient evaluation metric. These functions are found in the Dice_functions.py file, which is required as an import into the Group-UNetModel-LGG.py file. After the final version of the model is saved, the model can then be loaded into the LGG-UNetModel-Eval.py file. This file loads in the chosen LGG model and evaluates it on the training and validation datasets, returning the Dice scores per tumour region, confusion matrices, and example prediction images. Again, the Dice score functions are used for the evaluation.
Then, once the best group model has been determined, the model can be applied to the LGG test data. The LGG-Test.py file first applies all the same pre-processing steps to the test data, then loads in the chosen model and predicts on the test data. 

#### Individual model
In addition to the group model, the pre-processed LGG data can also be applied to the individual model, contained in file Individual-UNetModel-LGG.py. The individual U-Net file contains the architecture based on the original Ronneberger et al. U-Net, as well as all the optional additional pairs of 2D convolutional layers. These additional pairs were added one at a time, after each pair was added the model was re-evaluated to see if there had been an improvement in the Dice scores. To evaluate a model saved from the individual model file, the LGG-UNetModel-Eval.py file can be used again to load the model and return the training and validation data results. The LGG-Test.py file can also be used to load the final best model and apply the test data.

## HGG data
Similar to the LGG data, the HGG data is first read in and pre-processed through the file DataSplitHGG.py. This DataSplitHGG.py file reads in the HGG data, splits it into train, validate, and test sets, and then performs all the same pre-processing functions as on LGG. The pre-processed train and validate data sets are saved, and the unprocessed test set is saved.

#### Group model
After the HGG data has been pre-processed, the group U-Net can be trained on the HGG data. The file containing the group model for the HGG data is Group-UNetModel-HGG.py. It is the same as the model for the LGG, but instead reads in the HGG data and is only trained for 5 epochs. An additional file for the HGG data is needed to complete the training of the model, as the model is only able to train for 5 epochs at a time. For this reason, the HGG-Model-MoreEpochs.py file is used to read in the saved model and continue the training for another 5 epochs with the same loss function, evalutation metric, and data. To reach 20 epochs, for example, the model must be iterated through the HGG-Model-MoreEpochs.py file 3 additional times to reach 20. Once the final best model has been achieved, the HGG-UNetModel-Eval.py file and HGG-Test.py file can be used to apply the HGG train, validate, and test data to the model. The HGG-Tst.py file first pre-processes the HGG test data and then predicts with the model.

#### Individual model
Similar to the LGG data, the HGG data can also be applied to the individual model. The file to train on the HGG data is Individual-UNetModel-HGG.py. The architecture is the same as the one presented for the LGG data. This individual model is also only able to train for 5 epochs at a time, so the HGG-Model-MoreEpochs.py file is also used here to load and continuing training. The HGG-UNetModel-Eval.py file and HGG-Test.py files can be then used to load the final model and return the Dice scores.


## References
The code for the DataSplit.py, DataLoad.py, DataSplitHGG.py, Group-UNetModel-LGG.py, Group-UNetModel-HGG.py, and Preprocessing.py was all adapted from the Multimodal Brain Tumour
Segmentation GitHub repository by Aryaman Sinha (accessed July 1, 2020). The link to the repository can be found here: https://github.com/as791/Multimodal-Brain-Tumor-Segmentation/blob/master/Main_file.ipynb.

The code for the Individual-UNetModel-LGG.py and Individual-UNetModel-HGG.py files was based on the original Ronneberger et al. U-Net design [1], and the code is adapted from the Multimodal Brain Tumour Segmentation GitHub repository by Aryaman Sinha (accessed July 1, 2020) as well as from the UNET-TGS GitHub repository by Harshall Lamba (accessed July 20, 2020, from: https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb).

The code for the Dice_functions.py was adapted from the brats_2019 GitHub repository, which can be found at: https://github.com/woodywff/brats_2019/blob/master/demo_task1/evaluate.py (accessed July 3, 2020).

[1] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. CoRR. 2015;abs/1505.04597.  Available from:http://arxiv.org/abs/1505.04597.
