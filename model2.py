'''
#  model.py
#
#  Implements DCGAN model
#  Performs inpainting of masked images
#
#  Version date: 12/15/2018
'''

import os
from glob import glob
import tensorflow as tf
import time
import math
import itertools
#from utils2 import *
from model_funcs import *
from utils import block_mask

GENERATOR_F = 64
DISCRIMINATOR_F = 64

GENERATOR_FULLY_CONNECTED = 1024
DISCRIMINATOR_FULLY_CONNECTED = 1024

class DCGAN(object):
    def __init__(self, sess, 
                 batch_size=64,
                 sample_size=64,
                 z_dim=100,
                 checkpoint_dir=None,
                 lamda=0.1,
                 dataset = 'celeba'):
        """
        DCGAN Class:
        Builds the discriminator and generator networks, and stores them in self.G and self.D
        Contains a training method which trains the generator and discriminator.
        
        ARGUMENTS:
        - sess: tensorflow session
        - batch_size: batch size to use during training
        - sample_size: width/height of the input pictures (hardcoded at 64 pixels by 64 pixels)
        - z_dim: length of the z vector that is fed into the generator to create an image
        - checkpoint_dir: directory to store the checkpoint
        - lamda: relative strength of the prior loss versus the context loss
        - dataset: dataset to train the DCGAN on. Either 'celeba', 'cars', or 'svhn'; currently
        only 'celeba' works. This (preprocessed, cfr model_funcs.py) dataset has to be stored in
        ./datasets/img_align_celeba_preprocessed/.
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = 64
        self.sample_size = sample_size
        self.image_shape = [64, 64, 3]
        self.dataset = dataset
        self.z_dim = z_dim
        self.lamda = lamda


        # creates batch norm layers
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]

        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i,)) for i in range(6)]

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        '''
        This function builds the initial model to be trained on the datasets. Both the Discriminator and Generator are created.  
        Sigmoid Cross Entropy loss is used as the loss function.
        '''
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        
        # create Generator and Discriminators
        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # loss functions
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits,
                labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_,
                labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_,
                labels=tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        # save values 
        # SAVER NOT FULLY WORKING
        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lamda * self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)
        self.capped_gradient = tf.clip_by_value(self.grad_complete_loss, -1., 1.)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
        #self.train_op = optimizer.apply_gradients([(self.capped_gradient, self.z)])

    def train(self, config):
        '''
        Function which trains the generator and discriminator
        '''
        
        # perform optimization
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(
                self.d_loss,
                var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(
                self.g_loss,
                var_list=self.g_vars)                
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , 100))
        counter = 1
        start_time = time.time()
        
        # attempt to load checkpoint
        if self.load(self.checkpoint_dir):
            print('Existing checkpoint found')
        else:
            print('Starting training from scratch')
        batches, batch_idxs = get_batches(self.batch_size, self.dataset)

        for epoch in range(config.epoch):
            print('EPOCHS: ', epoch, ' / ', config.epoch)

            for idx in range(0, batch_idxs):
                batch_images = next(batches).astype(np.float32) / 255 * 2 - 1
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # apply gradients to the discriminator
                d_optim_ = self.sess.run(
                        d_optim,
                        feed_dict={self.images: batch_images, self.z: batch_z, self.is_training: True})

                # apply gradients to the generator
                g_optim_ = self.sess.run(
                        g_optim,
                        feed_dict={self.z: batch_z, self.is_training: True})

                # g_optim is often run twice, to prevent the discriminator error to go to zero too fast.
                g_optim_ = self.sess.run(
                        g_optim,
                        feed_dict={self.z: batch_z, self.is_training: True})
                
                # Calculate the losses to print them to the screen
                error_G = self.g_loss.eval({self.z: batch_z, self.is_training: False})
                error_D_1 = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                error_D_2 = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})

                counter += 1
                # print out status every n number of iterations
                if np.mod(counter, 1) == 0 or counter < 3:
                    print("epoch: {:2d}/{:2d} || iteration: {:4d}/{:4d} || time: {:4.4f} || discriminator loss: {:.8f} || generator loss: {:.8f}".format(
                        epoch,
                        config.epoch,
                        idx,
                        batch_idxs,
                        time.time() - start_time,
                        error_D_1 + error_D_2,
                        error_G))

                # save samples every 100 iterations, to observe the improvements of the generator
                # to generate realistic faces
                if np.mod(counter, 100) == 0:
                    samples = self.sess.run(self.G, feed_dict={self.z: sample_z, self.is_training: False})
                    save_image(samples[0], [1, 1],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(0, 0))

                # save checkpoint every 400 iterations
                if np.mod(counter, 400) == 2:
                    self.save(config.checkpoint_dir, counter)


    def discriminator(self, image, reuse=False):
        '''
        This function creates a Discriminator network, which takes an input image and returns the probability of if it is real
        or fake. Functions from model_funcs are used.
        Input Args:
        image - tensor of shape [batch_size, 64, 64, 3]
        Returns:
        h4 - tensor of shape [64, 2]
        '''
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, DISCRIMINATOR_F, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, DISCRIMINATOR_F * 2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, DISCRIMINATOR_F * 4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, DISCRIMINATOR_F * 8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
            return tf.nn.sigmoid(h4), h4


    def generator(self, z, reuse=False):
        '''
        This function creates a Generator network, which takes an input matrix and returns a generated image. 
        Functions from model_funcs are used.
        Input Args:
        z - tensor of shape [None, 100]
        Returns:
        output - tensor of shape [None, 64, 64, 3]
        '''
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            self.z_, self.h0_w, self.h0_b = linear(z, GENERATOR_F * 16 * 4 * 4, 'g_h0_lin', with_w=True)
    
            var_ = tf.reshape(self.z_, [-1, 4, 4, GENERATOR_F * 16])
            out_ = tf.nn.relu(self.g_bns[0](var_, self.is_training))

            out1, _, _ = conv2d_transpose(
                    out_,
                    [self.batch_size, 8, 8, GENERATOR_F * 8],
                    name='g_h1',
                    with_w=True
                    )
            out1 = tf.nn.relu(self.g_bns[1](out1, self.is_training))

            out2, _, _ = conv2d_transpose(
                    out1,
                    [self.batch_size, 16, 16, GENERATOR_F * 4],
                    name='g_h2',
                    with_w=True
                    )
            out2 = tf.nn.relu(self.g_bns[2](out2, self.is_training))

            out3, _, _ = conv2d_transpose(
                    out2,
                    [self.batch_size, 32, 32, GENERATOR_F * 2],
                    name='g_h3',
                    with_w=True
                    )
            out3 = tf.nn.relu(self.g_bns[3](out3, self.is_training))

            out4, _, _ = conv2d_transpose(
                    out3,
                    [self.batch_size, 64, 64, 3],
                    name='g_h4',
                    with_w=True
                    )
            return tf.nn.tanh(out4)

    def inpainting(self,
            test_image,
            iterations,
            mask_choice='block_mask',
            lamda=0.002):
        '''
        Test of Semantic inpainting
        this function applies a mask
        to the input image and then produces a visually similar image to the original

        uses functions from utils
        Input Arguments
        test- a single test image
        Outputs
        outputs- predicted images to match masked images. traverses a manifold using back-propogation
        '''

        #apply mask to image and keep mask for later use
        #self.image = test_image

        # MAKE SURE SELF.IMAGES HAS THE CORRECT SHAPE

        if mask_choice == 'block_mask':
            masked_test, mask = block_mask(test_image,30)
        elif mask_choice == 'random_mask':
            masked_test, mask = random_mask(test_image,0.6)
        elif mask_choice == 'half_missing_mask':
            masked_test, mask = half_missing_mask(test_image)
        else:
            print('incorrect mask choice')

        #reshape images and masks to be compatible with output from generator
        test_image = np.reshape(test_image,(1,64,64,3))
        mask = np.reshape(mask,(1,64,64,3))
        masked_test = np.reshape(masked_test,(1,64,64,3))

        #generate weights for contextual loss
        weight = np.zeros_like(mask)
        n = weight.shape[1]
        for i in range(n):
            for j in range(n):
                if (j-4) > 0 and (j+4) < (n - 4) and (i-4) >0 and i+4 < (n - 4) and mask[0,i,j,0] ==1:
                    cumulative_sum = 0;
                    for k in range(-3,3):
                        for l in range(-3,3):
                            if mask[0,i+k,l+j,0] ==0 and l!=0 and k!=0:
                                cumulative_sum = cumulative_sum + 1
                    cumulative_sum = cumulative_sum/49
                    weight[:,i,j,:] = cumulative_sum

        self.batch_size = 1
        z = np.random.uniform(-1, 1, [1, 100]).astype(np.float32)
        tf.global_variables_initializer().run()
        for i in range(iterations):
            fd = {
                self.mask: weight[0],
                self.z: z,
                self.images: test_image,
                self.is_training: False
            }
            run = [self.complete_loss, self.capped_gradient]
            loss, g = self.sess.run(run, feed_dict=fd)
            z = z - g[0]*learning_rate

        #rescale image Gz properly
        Gz = ((Gz + 1) / 2) * 255
        #crop out center and add it to test image
        fill = tf.multiply(tf.ones_like(self.mask) - self.mask,Gz)
        new_image =  masked_test + fill
        return new_image

    def save(self, checkpoint_dir, step):
        # saves checkpoint for later
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, 'DCGAN.model'),
                        global_step=step)

    def load(self, checkpoint_dir):
        # Loads pre-saved checkpoint
        checkpoint_dir = './checkpoint/'
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
