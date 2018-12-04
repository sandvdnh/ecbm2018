# This python script contains various functions for building a DCGAN

import numpy as np 
import tensorflow as tf


def leaky_relu(x, leak=0.2, name=''):
    #leaky Relu returns max of x and x*leak
    return tf.math.maximum(x, x * leak, name=name)

