import time
import numpy as np
import tensorflow as tf
from model import *

# For pre-loading data if first time using the dataset
my_iterator, _ = get_batches(10, dataset = 'celeba')
for i in range(3):
    images = next(my_iterator)

with tf.Session() as sess:
    #model.build_model()
    model = DCGAN(sess, input_height=64, input_width=64, batch_size=64, sample_num = 64, output_height=64, output_width=64,
            g_dim=[1024, 512, 256, 128], d_dim=[64, 128, 256, 512], s_size=4, z_dim=100, dataset='cars')
    model.train(epochs=2, batch_size=64, learning_rate=0.0002, beta1=0.5)

