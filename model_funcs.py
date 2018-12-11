# This python script contains various functions for building a DCGAN
import glob
import zipfile
import tarfile
from urllib import request
import os
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
    return tf.maximum(x, x * leak, name=name)



def get_batches(batch_size, dataset):
    '''
    Yields the correct iterator for each dataset.
    '''
    if dataset == 'svhn':
        return get_batches_svhn(batch_size), int(73257 / batch_size)
    elif dataset == 'celeba':
        return get_batches_celeba(batch_size)
    elif dataset == 'cars':
        return get_batches_cars(batch_size)
    else: 
        print('I do not know this dataset!')


def get_batches_svhn(batch_size):
    '''
    Loads svhn dataset and returns iterator.

    The function expects two data files to be present in ./datasets/:
    - train_32x32.mat (73257 images)
    - test_32x32.mat (26032 images)

    When first running this function, set load = True such that resized 64x64 images are saved to .npy files.
    '''
    filenames = [
            'train_32x32.mat',
            'test_32x32.mat'
            ]
    url = 'http://ufldl.stanford.edu/housenumbers/'
    directory = './datasets/'
    files = list(glob.glob(os.path.join(directory,'*.*')))
    for file_ in filenames:
        if directory + file_ not in files:
            print('Downloading ', file_)
            request.urlretrieve(
                    url + file_,
                    directory + file_
                    )

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
    directory = './datasets/'
    list_dirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
    filenames = [
            'img_align_celeba.zip',
            ]
    url = 'https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing'
    files = list(glob.glob(os.path.join(directory,'*.*')))
    if directory + 'img_align_celeba' not in list_dirs:
        for file_ in filenames:
            if directory + file_ not in files:
                print('Dataset not found; download .zip file from')
                print(url)
                print('and extract to ./datasets/img_align_celeba/*')
                return 0
            else:
                print('Extract .zip file to ./datasets/img_align_celeba/*')
    else:
        print('Dataset already present!')

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


def get_batches_cars(batch_size):
    '''
    returns iterator for the cars dataset
    Expects data files to be located in ./dataset/cars_train/
    '''
    filenames = [
            'cars_test.tgz',
            'cars_train.tgz',
            'car_devkit.tgz'
            ]
    url = 'http://imagenet.stanford.edu/internal/car196/'
    directory = './datasets/'
    files = list(glob.glob(os.path.join(directory,'*.*')))
    for file_ in filenames[:-1]:
        if directory + file_ not in files:
            print('Downloading ', file_)
            request.urlretrieve(
                    url + file_,
                    directory + file_
                    )
    if directory + filenames[-1] not in files:
        print('Downloading ', filenames[-1])
        url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
        request.urlretrieve(
                url,
                directory + filenames[-1]
                )

    list_dirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
    for file_ in filenames[:-1]:
        if directory + file_[:-4] not in list_dirs:
            print('Extracting ', file_)
            tar = tarfile.open(directory + file_, "r:gz")
            tar.extractall(path = directory)
            tar.close()
    if directory + 'devkit' not in list_dirs:
        print('Extracting ', filenames[-1])
        tar = tarfile.open(directory + filenames[-1], "r:gz")
        tar.extractall(path = directory)
        tar.close()

    path = './datasets/cars_train/'
    filenames = []
    filenames += [f for f in listdir(path) if isfile(join(path, f))]
    n = len(filenames)
    n_batches = int(n / batch_size)
    counter = 0
    bounding_boxes_ = loadmat('./datasets/devkit/cars_train_annos.mat')['annotations'][0]
    bounding_boxes = np.zeros((n, 4), dtype = np.uint)

    for i in range(len(bounding_boxes_)):
        minx = int(bounding_boxes_[i][0][0])
        miny = int(bounding_boxes_[i][1][0])
        maxx = int(bounding_boxes_[i][3][0])
        maxy = int(bounding_boxes_[i][2][0])
        bounding_boxes[i] = np.array([minx, maxx, miny, maxy])
        filenames[i] = bounding_boxes_[i][-1][0]

    while True and counter < 3:
        for i in range(n_batches):
            batch = np.zeros((batch_size, 64, 64, 3))
            for j in range(batch_size):
                x = plt.imread(path + filenames[i * batch_size + j])
                if len(x.shape) == 2:
                    x = np.repeat(np.reshape(x, (*x.shape, 1)), 3, axis = 2)
                x_total, y_total = x.shape[:2]
                minx = int(x_total) - int(bounding_boxes[i * batch_size + j, 0])
                maxx = int(x_total) - int(bounding_boxes[i * batch_size + j, 1])
                minx = bounding_boxes[i * batch_size + j, 0]
                maxy = bounding_boxes[i * batch_size + j, 1]
                miny = bounding_boxes[i * batch_size + j, 2]
                maxx = bounding_boxes[i * batch_size + j, 3]
                x = x[miny : maxy, minx : maxx, :].astype(np.uint)
                x = imresize(x, (64, 64, 3)) / 255
                x = 2 * x - 1
                batch[j] = x.copy()
            yield batch


# Simple example on how to use this
if __name__ == '__main__':
    my_iterator = get_batches(10, dataset = 'cars')
    for i in range(3):
        images = next(my_iterator)


