'''
#  model.py
#
#  Implement DCGAN model
'''

import tensorflow as tf
import numpy as np

from model_funcs import *
      
        
class DCGAN:
    def __init__(self, input_height=64, input_width=64,batch_size=64, sample_num = 64, 
         output_height=64, output_width=64,g_dim=[1024, 512, 256, 128], d_dim=[64, 128, 256, 512], s_size=4,):
        '''
        Initialize variables for implementation and calculations
        Args:
            batch_size: The size of the batch
            g_dim: Dimensions of generator filters in conv layers, as an array
            d_dim: Dimensions of discriminator filters in conv layers, as an array
            
        '''
        self.batch_size = batch_size
        self.sample_num = sample_num
        
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.g_dim = g_dim + [3]
        self.d_dim = [3] + d_dim

        self.s_size = s_size
        
        self.loss()
        self.generator()
        self.saver = tf.train.Saver()

        
    def generator(self, inputs, training=):
        '''
        Implementation of discriminator
        
        Input arguments:
        inputs - image 
        
        Outputs:
        
        '''
        
        with tf.variable_scope('generator'):
            #reshape from inputs
            reshape_out = tf.layers.dense(inputs, self.g_dim[0] * self.s_size * self.s_size)
            reshape_out = tf.reshape(reshape_out, [-1, self.s_size, self.s_size, self.g_dim[0]])
            reshape_out = tf.nn.relu(tf.layers.batch_normalization(reshape_out, training=training), name='outputs')
            
            # deconv layer 1
            conv_1 = tf.layers.conv2d_transpose(reshape_out, self.g_dim[1], [5, 5], strides=(2, 2), padding='SAME')
            conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=training), name='outputs')
            
            # deconv layer 2
            conv2 = tf.layers.conv2d_transpose(conv_1, self.g_dim[2], [5, 5], strides=(2, 2), padding='SAME')
            conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=training), name='outputs')
            
            # deconv layer 3
            conv3 = tf.layers.conv2d_transpose(conv_2, self.g_dim[3], [5, 5], strides=(2, 2), padding='SAME')
            conv_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=training), name='outputs')
            
            # deconv layer 4
            conv_4 = tf.layers.conv2d_transpose(outputs, self.g_dim[4], [5, 5], strides=(2, 2), padding='SAME')
            
            #output images
            outputs = tf.tanh(conv_4, name='outputs')
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return outputs
        
    def discriminator(self, inputs, training=):
        '''
        Implementation of discriminator
        Uses functions from model_funcs.py
        
        Input arguments:
        inputs
        
        Outputs:
        
        '''
        inputs = tf.convert_to_tensor(inputs)
        
        with tf.variable_scope('discriminator') as scope:
            # conv layer 1
            conv_1 = tf.layers.conv2d(inputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
            conv_1 = leaky_relu(tf.layers.batch_normalization(conv_1, training=training), name='outputs')
            
            # conv layer 2
            conv_2 = tf.layers.conv2d(conv_1, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
            conv_2 = leaky_relu(tf.layers.batch_normalization(conv_2, training=training), name='outputs')
            
            # conv layer 3
            conv_3 = tf.layers.conv2d(conv_2, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
            conv_3 = leaky_relu(tf.layers.batch_normalization(conv_3, training=training), name='outputs')
            
            # conv layer 4
            outputs = tf.layers.conv2d(conv_3, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            
            #reshape output
            batch_size = outputs.get_shape()[0].value
            reshape = tf.reshape(outputs, [batch_size, -1])
            outputs = tf.layers.dense(reshape, 2, name='outputs')
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return outputs
        
    def loss(self):
        '''
        Calculating loss based on 
        '''
        # One-hot coding
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
        
        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)
        
        
    def train(self, batches, learning_rate=0.0002, beta1=0.5):
        '''
        Input Args:
        batches - 
        max_count - 
        save_every_n - 
        
        Outputs:
        
        '''
        
        # using Adam optimizer as specified in the project paper
        d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1) #generator optimizer
        g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1) # discriminator optimizer
        
        tf.global_variables_initializer().run()
        
        counter = 1
        start_time = time.time()
        
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")
        
        # learning variables
        num_train = 49000
        batch_size = 250
        num_batch = num_train//batch_size
        # run
        with tf.Session() as sess:
            sess.run(init)
            for e in range(20):
                for i in range(num_batch):
                    batch_xs, batch_ys = X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
                    sess.run(train_step, feed_dict={x_tf: batch_xs, y_tf: batch_ys})
                val_acc = sess.run(accuracy, feed_dict={x_tf: X_val, y_tf: y_val})
                print('epoch {}: valid acc = {}'.format(e+1, val_acc))
            # get test accuracy
            test_acc = sess.run(accuracy, feed_dict={x_tf: X_val, y_tf: y_val})
            print('test acc = {}'.format(test_acc))
        
        '''
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            counter = 0
            new_state = sess.run(self.initial_state)
            # Train network
            for x in batches:
                counter += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                _, g_loss_value , d_loss_value = sess.run([self.loss, 
                                                     self.final_state, 
                                                     self.optimizer], 
                                                     feed_dict=feed)
                    
                end = time.time()
                if counter % 200 == 0:
                    print('step: {} '.format(counter),
                          'loss: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
                    
                if (counter % save_every_n == 0):
                    self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.s_size))
                    
                if counter >= max_count:
                    break
            
            self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.s_size))
        '''
        
        