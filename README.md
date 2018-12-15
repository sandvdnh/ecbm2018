# Final project ECBM4040

The code can be run from either the jupyter notebook 'Project_Notebook.ipynb' or the standalone script 'standalone.py'.
Both of them import helper functions from ops.py and model_funcs.py, which take care of data loading/batch generation and layer operations like linear/convolutions and batch norm.

The code is meant to be agnostic towards the specific dataset (CelebA, Cars, or SVHN), and model_funcs.py contains methods to generate batches from all three datasets.
However, we only tested it on the CelebA dataset. To get it working on either cars or svhn, it might be necessary to change the preprocessing procedure in model_funcs.py
because we currently run into memory errors when loading all data files at once (even on google cloud).


## Download the dataset

The CelebA dataset can be downloaded from https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing
and should be extracted into './datasets/' (such that images are stored in './datasets/img_align_celeba/')
In ./datasets/, make a new folder called 'img_align_celeba_preprocessed'.
After that, run model_funcs.py, which will preprocess the images and save them in img_align_celeba_preprocessed, in order to speed up training (and to avoid memory errors).

This sets up the data. You can now run either the notebook or the standalone script to start training the DCGAN.
Periodically, it will write out checkpoint files to a checkpoint folder (which will be created automatically if it's not already there). 
Sample images during training (to evaluate the quality of the generator) are stored in the samples folder (which is also automatically created if it's not already there).


## Function of each file:

Inpainting_test.ipynb: Used to test inpainting function with un-trained generator

Project_Notebook.ipynb: Main function to train DCGAN and implement inpainting		

data_utils: functions for preprocessing data

model.py: contains all functions neccesary for running DCGAN and image inpainting

model_funcs: secondary functions used by model.py

utils: mask and crop functions are located her

model.py: contains all functions neccesary for running DCGAN and image inpainting


## Data Set locations:
Run Project_notebook.ipynb should automatically download the datasets and has a link to them.
Those links are also posted below:

celeba: https://www.kaggle.com/jessicali9530/celeba-dataset

SVHN: http://ufldl.stanford.edu/housenumbers/

cars: https://ai.stanford.edu/~jkrause/cars/car_dataset.html





