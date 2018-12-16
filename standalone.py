import os
import scipy.misc
import numpy as np
from model import DCGAN
from model_funcs import *
import tensorflow as tf

# prevent tensorflow from allocating too much memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

flags = tf.app.flags
# FOLDER SETTINGS
flags.DEFINE_string("dataset", "./datasets/img_align_celeba_preprocessed", "dir where preprocessed dataset is stored (see readme)")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "dir where checkpoints are saved")
flags.DEFINE_string("sample_dir", "samples", "dir where sample images from generator are saved during training")

# DATA SETTINGS
flags.DEFINE_integer("train_size", 202600, "dataset size (only valid for CelebA)")
flags.DEFINE_integer("batch_size", 64, "number of samples in batch")
flags.DEFINE_integer("image_size", 64, "fixed height of images")

# TRAINING SETTINGS
flags.DEFINE_integer("epoch", 1, "number of epochs to train")
flags.DEFINE_float("learning_rate", 0.0002, "learning rate in the adam optimizers")
flags.DEFINE_float("beta1", 0.5, "adam setting")
FLAGS = flags.FLAGS

# check if checkpoint and sample images folder exist, if not create them
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

with tf.Session(config=config) as sess:
    #create DC gan
    dcgan = DCGAN(
            sess,
            batch_size=FLAGS.batch_size,
            checkpoint_dir=FLAGS.checkpoint_dir)
    dcgan.train(FLAGS)
    #dcgan.inpainting(images[0], 5, mask_choice = 'block_mask')

