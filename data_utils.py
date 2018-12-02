import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from scipy.misc import imresize

'''
#  data_utils.py
#
#  Extrating the data for processing
'''

def load_svhn(load = False):
    '''
    Loads svhn dataset, which contains two files:
    - train_32x32.mat (73257 images)
    - test_32x32.mat (26032 images)

    When first running this function, set load = True such that resized 64x64 images are saved to .npy files.
    '''
    if load:
        # Load data from .mat files, resize to 64x64
        loaded = loadmat('./datasets/train_32x32.mat')
        X_train = np.rollaxis(loaded['X'], 3)
        training_data = np.zeros((73257, 64, 64, 3))
        for i in range(X_train.shape[0]):
            training_data[i, :, :, :] = imresize(X_train[i], (64, 64, 3)) / 255
        # rescale each image to have pixel values in [-1, 1]
        training_data = 2 * training_data - 1
        np.save('./datasets/training_data.npy', training_data)
        
        loaded = loadmat('./datasets/test_32x32.mat')
        X_test = np.rollaxis(loaded['X'], 3)
        testing_data = np.zeros((26032, 64, 64, 3))
        for i in range(X_test.shape[0]):
            testing_data[i, :, :, :] = imresize(X_test[i], (64, 64, 3)) / 255
        # rescale each image to have pixel values in [-1, 1]
        testing_data = 2 * testing_data - 1
        np.save('./datasets/testing_data.npy', testing_data)

    else:
        testing_data = np.load('./datasets/testing_data.npy')
        training_data = np.load('./datasets/training_data.npy')
    return training_data, testing_data

def load_celebs():
    '''
    Loads celebs dataset
    UNFINISHED
    '''
    filenames = []
    path = './datasets/img_align_celeba/'
    #for i in range(len(listdir(path))):
    for i in range(10):
        filenames += [f for f in listdir(path) if isfile(join(path, f))]
        x = np.expand_dims(plt.imread(path + filenames[i]), 0).astype(np.float32)
        print(x.shape)


    return 0

load_celebs()
