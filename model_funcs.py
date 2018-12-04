# This python script contains various functions for building a DCGAN
from os import listdir
from os.path import isfile, join
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import imresize
from utils import crop

try:
    merge_summary = tf.merge_summary    
except:
    merge_summary = tf.summary.merge

def leaky_relu(x, leak=0.2, name=''):
    #leaky Relu returns max of x and x*leak
    return tf.math.maximum(x, x * leak, name=name)


def get_batches(batch_size, dataset):
    '''
    Yields the correct iterator for each dataset.
    '''
    if dataset == 'svhn':
        return get_batches_svhn(batch_size)
    elif dataset == 'celeba':
        return get_batches_celeba(batch_size)
    elif dataset == 'cars':
        return get_batches_cars(batch_size)
    else: 
        print('I do not know this dataset!')


def get_batches_svhn(batch_size, load = False):
    '''
    Loads svhn dataset and returns iterator.

    The function expects two data files to be present in ./datasets/:
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
        #testing_data = np.load('./datasets/testing_data.npy')
        training_data = np.load('./datasets/training_data.npy')

    # create batches
    n = training_data.shape[0]
    n_batches = int(n / batch_size)
    while True:
        for i in range(n_batches):
            batch = training_data[i * batch_size : (i + 1) * batch_size]
            yield batch


def get_batches_celeba(batch_size):
    '''
    Returns iterator for celebs dataset
    Expects data files to be located in ./datasets/img_align_celeba/
    '''
    path = './datasets/img_align_celeba/'
    filenames = []
    filenames += [f for f in listdir(path) if isfile(join(path, f))]
    n = len(filenames)
    n_batches = int(n / batch_size)
    while True:
        for i in range(n_batches):
            batch = np.zeros((batch_size, 64, 64, 3))
            for j in range(batch_size):
                x = plt.imread(path + filenames[i * batch_size + j])
                x = imresize(x, 60) / 255
                x = crop(x)
                x = 2 * x - 1
                batch[j] = x.copy()
            yield batch


# Simple example on how to use this
if __name__ == '__main__':
    my_iterator = get_batches(10, dataset = 'celeba')
    for i in range(3):
        images = next(my_iterator)

