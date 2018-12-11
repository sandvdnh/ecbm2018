'''
#  model.py
#
#  Implement DCGAN model
'''

import time
import tensorflow as tf
import numpy as np

from model_funcs import *
  
    
class DCGAN:
    def __init__(self, sess, input_height=64, input_width=64, batch_size=64, sample_num = 64, output_height=64, output_width=64,
                 g_dim=[1024, 512, 256, 128], d_dim=[64, 128, 256, 512], s_size=4, z_dim=100, dataset='SVHN'):
        '''
        Initialize variables for implementation and calculations
        Args:
            batch_size: The size of the batch
            g_dim: Dimensions of generator filters in conv layers, as an array
            d_dim: Dimensions of discriminator filters in conv layers, as an array
            
        '''
        self.sess = sess
        self.batch_size = batch_size
        self.sample_num = sample_num
        
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.g_dim = g_dim + [3]
        self.d_dim = [3] + d_dim

        self.s_size = s_size
        self.z_dim = z_dim
        
        self.dataset = dataset
        
        self.reuse = False
        
        
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)
        
        self.build_model()
        
    
    
        
    def generator(self, inputs, training=False):
        '''
        Implementation of discriminator
        
        Input arguments:
        inputs - image 
        
        Outputs:
        
        '''
        
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            #reshape from inputs
            reshape_out = tf.layers.dense(inputs, self.g_dim[0] * self.s_size * self.s_size)
            reshape_out = tf.reshape(reshape_out, [-1, self.s_size, self.s_size, self.g_dim[0]])
            reshape_out = tf.nn.relu(tf.layers.batch_normalization(reshape_out, training=training), name='g_reshape')
            
            # deconv layer 1
            conv_1 = tf.layers.conv2d_transpose(reshape_out, self.g_dim[1], [5, 5], strides=(2, 2), padding='SAME')
            conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=training), name='g_conv_1')
            
            # deconv layer 2
            conv_2 = tf.layers.conv2d_transpose(conv_1, self.g_dim[2], [5, 5], strides=(2, 2), padding='SAME')
            conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=training), name='g_conv_2')
            
            # deconv layer 3
            conv_3 = tf.layers.conv2d_transpose(conv_2, self.g_dim[3], [5, 5], strides=(2, 2), padding='SAME')
            conv_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=training), name='g_conv_3')
            
            # deconv layer 4
            conv_4 = tf.layers.conv2d_transpose(conv_3, self.g_dim[4], [5, 5], strides=(2, 2), padding='SAME')
            
            #output images
            outputs = tf.tanh(conv_4, name='g_outputs') # Output shape is [batch_size (64), height (64), width (64), channels (3)]
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return outputs
        
    def discriminator(self, inputs, training=False):
        '''
        Implementation of discriminator
        Uses functions from model_funcs.py
        
        Input arguments:
        inputs
        
        Outputs:
        
        '''
        inputs = tf.convert_to_tensor(inputs)
        
        #with tf.variable_scope('discriminator') as scope:
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
            # conv layer 1
            conv_1 = tf.layers.conv2d(inputs, self.d_dim[1], [5, 5], strides=(2, 2), padding='SAME')
            conv_1 = leaky_relu(tf.layers.batch_normalization(conv_1, training=training), name='d_conv_1')
            
            # conv layer 2
            conv_2 = tf.layers.conv2d(conv_1, self.d_dim[2], [5, 5], strides=(2, 2), padding='SAME')
            conv_2 = leaky_relu(tf.layers.batch_normalization(conv_2, training=training), name='d_conv_2')
            
            # conv layer 3
            conv_3 = tf.layers.conv2d(conv_2, self.d_dim[3], [5, 5], strides=(2, 2), padding='SAME')
            conv_3 = leaky_relu(tf.layers.batch_normalization(conv_3, training=training), name='d_conv_3')
            
            # conv layer 4
            conv_4 = tf.layers.conv2d(conv_3, self.d_dim[4], [5, 5], strides=(2, 2), padding='SAME')
            conv_4 = leaky_relu(tf.layers.batch_normalization(conv_4, training=training), name='d_conv_4')
            
            #reshape output
            batch_size = conv_4.get_shape()[0].value
            reshape = tf.reshape(conv_4, [batch_size, -1])
            outputs = tf.layers.dense(reshape, 2, name='d_outputs') # outputs shape is [batch_size, 2]
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return outputs
         
    def build_model(self):
        '''
        build model, calculate losses
        '''
        # prep values to build model
        image_dims = [self.input_height, self.input_width, 3]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs # tensor of shape [batch, height, width, channels]
        
        # build models
        generated = self.generator(self.z , training=True)
        g_outputs = self.discriminator(generated, training=True)
        t_outputs = self.discriminator(inputs, training=True)
        
        # Softmax cross entropy loss
        #g_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=g_outputs, labels=tf.zeros([self.batch_size]))
        g_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)
        self.g_loss = tf.reduce_mean(g_loss)
        
        #d_loss_real = tf.nn.softmax_cross_entropy_with_logits_v2(logits=t_outputs, labels=tf.ones([self.batch_size]))
        d_loss_real = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=t_outputs)
        self.d_loss_real = tf.reduce_mean(d_loss_real)
        
        #d_loss_fake = tf.nn.softmax_cross_entropy_with_logits_v2(logits=g_outputs, labels=tf.zeros([self.batch_size]))
        d_loss_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)
        self.d_loss_fake = tf.reduce_mean(d_loss_fake)
        
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
    
    def train(self, epochs=2, batch_size=245, learning_rate=0.0002, beta1=0.5):
        '''
        Input Args:
        batches - batches of images for training
        epochs
        batch_size
        save_every_n - 
        
        Outputs:
        
        '''
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        
        with tf.variable_scope('scope', reuse = tf.AUTO_REUSE ):
            # using Adam optimizer as specified in the project paper
            d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.d_loss) #generator optimizer
            g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.g_loss) # discriminator optimizer

        tf.global_variables_initializer().run()
        
        counter = 1
        start_time = time.time()
        
        '''
        # For loading a checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")
        '''
        # run
        #with tf.Session() as sess:
           # sess.run(init)
            
        #generate batches based on which dataset is being used
        #if self.data_set == '': --> uncomment if needed
        print("loading data.... ")
        batches, iters = get_batches(batch_size, self.dataset)
        
        # getting number of training iterations
        # iters = 1000
        print('number of batches for training: {}'.format(iters))

        for epoch in range(epochs):
            
            for i in range(iters):
                #get batch
                batch_images = next(batches)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                #_, g_loss_value, d_loss_value = sess.run([train_op, losses[self.g], losses[self.d]])

                # ---> Run g_optim twice to make sure that d_loss does not go to zero?
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.inputs: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # ---> Run g_optim twice to make sure that d_loss does not go to zero (different from paper)?
                '''
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)
                '''

                errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, config.epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake+errD_real, errG))

                '''
                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_inputs,
                        },
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                            './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                    except:
                        print("one pic error!...")
                '''

                # Save checkpoint
                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

        