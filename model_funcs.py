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
from scipy.misc import imresize, imsave
from utils import crop
import math
from utils import *
from tensorflow.python.framework import ops


try:
    merge_summary = tf.merge_summary    
except:
    merge_summary = tf.summary.merge

#def leaky_relu(x, leak=0.2, name=''):
#    #leaky Relu returns max of x and x*leak
#    return tf.maximum(x, x * leak, name=name)
#
#def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
#    shape = input_.get_shape().as_list()
#    print('linear shape = ',shape)
#    with tf.variable_scope(scope or "Linear", reuse=tf.AUTO_REUSE):
#        try:
#            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
#                 tf.random_normal_initializer(stddev=stddev))
#        except ValueError as err:
#            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
#            err.args = err.args + (msg,)
#            raise
#        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
#        if with_w:
#            return tf.matmul(input_, matrix) + bias, matrix, bias
#        else:
#            return tf.matmul(input_, matrix) + bias  


def get_batches(batch_size, dataset, first_time = False):
    '''
    Yields the correct iterator for each dataset.
    '''
    if dataset == 'svhn':
        return get_batches_svhn(batch_size), int(73257 / batch_size)
    elif dataset == 'celeba' and first_time:
        return get_batches_celeba(batch_size, use_preprocessed = True, preprocessed = False), int(202600 / batch_size)
    elif dataset == 'celeba' and not first_time:
        return get_batches_celeba(batch_size, use_preprocessed = True, preprocessed = True), int(202600 / batch_size)
    elif dataset == 'cars':
        return get_batches_cars(batch_size), int(8144 / batch_size)
    else: 
        print('I do not know this dataset!')


def get_batches_svhn(batch_size):
    '''
    Loads svhn dataset and returns iterator.

    The function expects two data files to be present in ./datasets/:
    - train_32x32.mat (73257 images)
    - test_32x32.mat (26032 images)
    If these are not present, they are downloaded automatically
    '''
    filenames = [
            'train_32x32.mat',
            'test_32x32.mat'
            ]
    url = 'http://ufldl.stanford.edu/housenumbers/'
    directory = 'datasets'
    files = list(glob.glob(os.path.join(directory,'*.*')))
    #print(os.path.join(directory, filenames[0]))
    for file_ in filenames:
        if os.path.join(directory, file_) not in files:
            print('Downloading ', file_)
            request.urlretrieve(
                    url + file_,
                    os.path.join(directory, file_)
                    )
    print('Data downloaded, preprocessing...')

    # Load data from .mat files, resize to 64x64
    loaded = loadmat('./datasets/train_32x32.mat')
    X_train = np.rollaxis(loaded['X'], 3)
    training_data = np.zeros((73257, 64, 64, 3))
    for i in range(X_train.shape[0]):
        training_data[i, :, :, :] = 2 * imresize(X_train[i], (64, 64, 3)) / 255 - 1
    # rescale each image to have pixel values in [-1, 1]
    #training_data = 2 * training_data - 1

    loaded = loadmat('./datasets/test_32x32.mat')
    X_test = np.rollaxis(loaded['X'], 3)
    testing_data = np.zeros((26032, 64, 64, 3))
    for i in range(X_test.shape[0]):
        testing_data[i, :, :, :] = 2 * imresize(X_test[i], (64, 64, 3)) / 255 - 1
    # rescale each image to have pixel values in [-1, 1]
    #testing_data = 2 * testing_data - 1

    # create batches
    n = training_data.shape[0]
    n_batches = int(n / batch_size)
    while True:
        for i in range(n_batches):
            batch = training_data[i * batch_size : (i + 1) * batch_size]
            yield batch


def get_batches_celeba(batch_size, preprocessed = True, use_preprocessed = True):
    '''
    Returns iterator for celebs dataset
    Expects data files to be located in ./datasets/img_align_celeba/
    '''
    directory = 'datasets'
    list_dirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
    filenames = [
            'img_align_celeba.zip',
            ]
    url = 'https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing'
    files = list(glob.glob(os.path.join(directory,'*.*')))
    if os.path.join(directory, 'img_align_celeba') not in list_dirs:
        for file_ in filenames:
            if os.path.join(directory, file_) not in files:
                print('Dataset not found; download .zip file from')
                print(url)
                print('and extract to ./datasets/img_align_celeba/*')
                return 0
            else:
                print('Extract .zip file to ./datasets/img_align_celeba/*')
    else:
        print('Dataset already present!')

    path = os.path.join('datasets', 'img_align_celeba')
    filenames = []
    filenames += [f for f in listdir(path) if isfile(join(path, f))]
    n = len(filenames)
    n_batches = int(n / batch_size)
    preprocessed_dir = os.path.join('datasets', 'img_align_celeba_preprocessed')
    if not preprocessed:
        print('preprocessing images and saving them... ')
        for i in range(n_batches):
            batch = np.zeros((batch_size, 64, 64, 3))
            print(i, n_batches)
            for j in range(batch_size):
                try:
                    x = plt.imread(os.path.join(path, filenames[i * batch_size + j]))
                    x = imresize(x, 60) / 255
                    x = crop(x)
                    x = 2 * x - 1
                    imsave(os.path.join(preprocessed_dir, '{:06d}'.format(i * batch_size + j) + '.jpg'), x)
                except OSError:
                    print('OSERROR')
    if use_preprocessed:
        filenames = []
        filenames += [f for f in listdir(preprocessed_dir) if isfile(join(preprocessed_dir, f))]
        while True:
            for i in range(n_batches):
                batch = np.zeros((batch_size, 64, 64, 3))
                for j in range(batch_size):
                    x = plt.imread(os.path.join(preprocessed_dir, filenames[i * batch_size + j]))
                    batch[j] = x.copy()
                yield batch
    else:
        while True:
            for i in range(n_batches):
                batch = np.zeros((batch_size, 64, 64, 3))
                for j in range(batch_size):
                    x = plt.imread(os.path.join(path, filenames[i * batch_size + j]))
                    x = imresize(x, 60) / 255
                    x = crop(x)
                    x = 2 * x - 1
                    batch[j] = x.copy()
                yield batch


def get_batches_cars(batch_size):
    '''
    returns iterator for the cars dataset
    Expects data files to be located in ./dataset/cars_train/
    8144 training images
    '''
    filenames = [
            'cars_test.tgz',
            'cars_train.tgz',
            'car_devkit.tgz'
            ]
    url = 'http://imagenet.stanford.edu/internal/car196/'
    directory = 'datasets'
    files = list(glob.glob(os.path.join(directory,'*.*')))
    for file_ in filenames[:-1]:
        if os.path.join(directory, file_) not in files:
            print('Downloading ', file_)
            request.urlretrieve(
                    url + file_,
                    os.path.join(directory, file_)
                    )
    if os.path.join(directory, filenames[-1]) not in files:
        print('Downloading ', filenames[-1])
        url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
        request.urlretrieve(
                url,
                os.path.join(directory, filenames[-1])
                )

    list_dirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
    for file_ in filenames[:-1]:
        if os.path.join(directory, file_[:-4]) not in list_dirs:
            print('Extracting ', file_)
            tar = tarfile.open(os.path.join(directory, file_), "r:gz")
            tar.extractall(path = directory)
            tar.close()
    if os.path.join(directory, 'devkit') not in list_dirs:
        print('Extracting ', filenames[-1])
        tar = tarfile.open(os.path.join(directory, filenames[-1]), "r:gz")
        tar.extractall(path = directory)
        tar.close()

    #path = './datasets/cars_train/'
    path = os.path.join('datasets', 'cars_train')
    filenames = []
    filenames += [f for f in listdir(path) if isfile(join(path, f))]
    n = len(filenames)
    n_batches = int(n / batch_size)
    counter = 0
    path_ = os.path.join('datasets', 'devkit', 'cars_train_annos.mat')
    bounding_boxes_ = loadmat(path_)['annotations'][0]
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
                x = plt.imread(os.path.join(path, filenames[i * batch_size + j]))
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


class batch_norm(object):
    '''
    Implementation of batch norm
    FROM STACK OVERFLOW
    '''
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.name = name

    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    '''
    Computes cross-entropy
    '''
    eps = 1e-12 ## parameter needed to ensure regular behavior
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv2d(
        input_,
        output_dim,
        height=5,
        width=5,
        height_=2,
        width_=2,
        std=0.02,
        name="conv2d"):
    '''
    conv2d op
    '''
    with tf.variable_scope(name):
        # Define weights
        w = tf.get_variable(
                'w',
                [height, width, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=std))
        # perform convolution
        conv = tf.nn.conv2d(
                input_,
                w,
                strides=[1, height_, width_, 1],
                padding='SAME')
        # biases
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def conv2d_transpose(
        input_,
        output_shape,
        height=5,
        width=5,
        height_=2,
        width_=2,
        std=0.02,
        name="conv2d_transpose",
        return_vars=False):
    '''
    transposed conv2d
    '''
    with tf.variable_scope(name):
        # Define weights
        w = tf.get_variable(
                'w',
                [height, width, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.random_normal_initializer(stddev=std))
        # perform 'deconvolution'
        deconv = tf.nn.conv2d_transpose(
                input_,
                w,
                output_shape=output_shape,
                strides=[1, height_, width_, 1])
        # biases
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        # return result and variables depending on the value of the boolean return_vars
        if return_vars:
            return deconv, w, biases
        else:
            return deconv

def lrelu(
        x,
        leak=0.2,
        name="lrelu"):
    '''
    leaky Rectified Linear Unit
    '''
    with tf.variable_scope(name):
        f_1 = 0.5 * (1 + leak)
        f_2 = 0.5 * (1 - leak)
        return f_1 * x + f_2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, return_vars=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

# Simple example on how to use this
if __name__ == '__main__':
    my_iterator, _ = get_batches(10, dataset = 'celeba', first_time = True)
    for i in range(20):
        images = next(my_iterator)
