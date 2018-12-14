# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf

from ops2 import *
from utils2 import *
from model_funcs import *

#SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
#
#def dataset_files(root):
#    """Returns a list of all image files in the given directory"""
#    return list(itertools.chain.from_iterable(
#        glob.glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


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
                 lam=0.1,
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
        - lam: relative strength of the prior loss versus the context loss
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

        #self.lowres = lowres
        #self.lowres_size = image_size // lowres
        #self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim

        #self.gf_dim = gf_dim
        #self.df_dim = df_dim

        #self.gfc_dim = gfc_dim
        #self.dfc_dim = dfc_dim

        self.lam = lam


        # creates batch norm layers
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]

        log_size = 6 # log_2(64)
        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        #self.model_name = "DCGAN.model"

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        #self.lowres_images = tf.reduce_mean(tf.reshape(self.images,
        #    [self.batch_size, self.lowres_size, self.lowres,
        #     self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        #self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
        #    [self.batch_size, self.lowres_size, self.lowres,
        #     self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        #self.lowres_mask = tf.placeholder(tf.float32, self.lowres_shape, name='lowres_mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
        #self.contextual_loss += tf.reduce_sum(
        #    tf.contrib.layers.flatten(
        #        tf.abs(tf.multiply(self.lowres_mask, self.lowres_G) - tf.multiply(self.lowres_mask, self.lowres_images))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, config):
        #data = dataset_files(config.dataset)
        #np.random.shuffle(data)
        #assert(len(data) > 0)

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

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , 100))
        #sample_files = data[0:self.sample_size]

        #sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        #sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print('Existing checkpoint found')
        else:
            print('Starting training from scratch')
        batches, batch_idxs = get_batches(self.batch_size, self.dataset)

        for epoch in range(config.epoch):
        #    data = dataset_files(config.dataset)
        #    batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in range(0, batch_idxs):
                #batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                #batch = [get_image(batch_file, self.image_size, is_crop=False)
                #         for batch_file in batch_files]
                #batch_images = np.array(batch).astype(np.float32)
                #print(np.mean(batch_images[0]))
                #print(batch_images[0])
                #print(np.min(batch_images[0]))
                #print(np.max(batch_images[0]))
                batch_images = next(batches).astype(np.float32) / 255 * 2 - 1
                #print(np.mean(batch_images[0]))
                #print(batch_images[0])
                #print(np.min(batch_images[0]))
                #print(np.max(batch_images[0]))

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 0:
                    samples = self.sess.run(
                        self.G,
                        feed_dict={self.z: sample_z, self.is_training: False}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(0, 0))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, DISCRIMINATOR_F, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, DISCRIMINATOR_F * 2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, DISCRIMINATOR_F * 4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, DISCRIMINATOR_F * 8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, GENERATOR_F * 8 * 4 * 4, 'g_h0_lin', with_w=True)
    
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, GENERATOR_F * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            out1, _, _ = conv2d_transpose(
                    hs[0],
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


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, 'DCGAN.model'),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False


    def complete(self, config):
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        #lowres_mask = np.zeros(self.lowres_shape)
        if config.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            assert(config.centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*config.centerScale)
            u = int(self.image_size*(1.0-config.centerScale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:,:c,:] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        elif config.maskType == 'grid':
            mask = np.zeros(self.image_shape)
            mask[::4,::4,:] = 1.0
        #elif config.maskType == 'lowres':
        #    lowres_mask = np.ones(self.lowres_shape)
        #    mask = np.zeros(self.image_shape)
        else:
            assert(False)

        for idx in range(0, batch_idxs):
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            # !!Change
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = min(8, batchSz)
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, mask)
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'masked.png'))
            for img in range(batchSz):
                with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                    f.write('iter loss ' +
                            ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) +
                            '\n')

            for i in range(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    #self.lowres_mask: lowres_mask,
                    self.images: batch_images,
                    self.is_training: False
                }
                #run = [self.complete_loss, self.grad_complete_loss, self.G, self.lowres_G]
                #loss, g, G_imgs, lowres_G_imgs = self.sess.run(run, feed_dict=fd)
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img+1])

                if i % config.outInterval == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    nRows = np.ceil(batchSz/8)
                    nCols = min(8, batchSz)
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)
                    #if lowres_mask.any():
                    #    imgName = imgName[:-4] + '.lowres.png'
                    #    save_images(np.repeat(np.repeat(lowres_G_imgs[:batchSz,:,:,:],
                    #                          self.lowres, 1), self.lowres, 2),
                    #                [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                    completed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

                if config.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                    v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - config.beta1 ** (i + 1))
                    v_hat = v / (1 - config.beta2 ** (i + 1))
                    zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                    zhats = np.clip(zhats, -1, 1)

                elif config.approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, self.z_dim)
                    v_old = np.copy(v)

                    for steps in range(config.hmcL):
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]
                        zhats += config.hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
                        logprob = config.hmcBeta * loss[img] + np.sum(v[img]**2)/2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    config.hmcBeta *= config.hmcAnneal

                else:
                    assert(False)

