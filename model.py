'''
#  model.py
#
#  Implement DCGAN model
'''

import time
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model_funcs import *
  
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
    
class DCGAN:
    def __init__(self, sess, input_height=64, input_width=64, batch_size=64, sample_num = 64, output_height=64, output_width=64,
                 g_dim=[1024, 512, 256, 128], d_dim=[64, 128, 256, 512], s_size=4, y_dim=None, z_dim=100, dataset='SVHN'):
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
        self.y_dim = y_dim
        self.z_dim = z_dim
        
        self.dataset = dataset
        
        self.reuse = False
        
        
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)
        self.z_sum = tf.summary.histogram("z", self.z)
        
        self.build_model()
        
    
    
    def inpainting(self,learning_rate,test_image,iterations,lamda=0.002):
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
        import numpy as np
        import tensorflow as tf
        from utils import block_mask
        from utils import random_mask
        from utils import half_missing_mask
        
        #apply mask to image and keep mask for later use
        self.image = test_image
        masked_test, mask = block_mask(self.image,30)
        test_image = np.reshape(test_image,(1,64,64,3))
        mask = np.reshape(mask,(1,64,64,3))
        masked_test = np.reshape(masked_test,(1,64,64,3))
        
        self.image = tf.convert_to_tensor(test_image, dtype=tf.float32)
        self.mask = tf.convert_to_tensor(mask,dtype=tf.float32)
        self.learning_rate = tf.convert_to_tensor(learning_rate,dtype=tf.float32)
        
        #self.mask = mask
        
        #generate random z as a changeable variable 
        #self.z = np.random.uniform(-1, 1, [1, 100]).astype(np.float32) 
        self.z = tf.random_uniform([1,100],minval=-1,maxval=1,dtype=tf.float32,seed=None,name='z')
       
        #Define loss as sum of both types of loss
        self.weighted_context_loss = tf.reduce_sum( tf.abs(tf.multiply(self.generator(self.z),self.mask) - tf.multiply(self.image,self.mask) ) )
        self.perceptual_loss = self.g_loss 
        self.complete_loss = self.weighted_context_loss + lamda*self.perceptual_loss
        
        #define optimization function (gradient descent)
        self.gradients = tf.gradients(self.complete_loss,self.z)
        
        #gradient descent back propogation to update input z
        tf.global_variables_initializer().run()
        for i in range(iterations):
            
            loss, g, Gz = self.sess.run([self.complete_loss,self.gradients,self.generator(self.z)])
           
            self.z = self.z - g[0]*learning_rate 
            
           
        #rescale image Gz properly
        Gz = ((Gz + 1) / 2) * 255
        #crop out center and add it to test image
        fill = tf.multiply(tf.ones_like(self.mask) - self.mask,Gz)
        new_image =  masked_test + fill
        
        return new_image  
        
        
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
        #g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=g_outputs, labels=tf.zeros([self.batch_size]))
        g_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self.batch_size], dtype=tf.int64), logits=g_outputs)
        self.g_loss = tf.reduce_mean(g_loss)
        
        #d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=t_outputs, labels=tf.ones([self.batch_size]))
        d_loss_real = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self.batch_size], dtype=tf.int64), logits=t_outputs)
        self.d_loss_real = tf.reduce_mean(d_loss_real)
        
        #d_loss_fake = tf.nn.softmax_cross_entropy_with_logits_v2(logits=g_outputs, labels=tf.zeros([self.batch_size]))
        d_loss_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([self.batch_size], dtype=tf.int64),logits=g_outputs)
        self.d_loss_fake = tf.reduce_mean(d_loss_fake)
        
        self.d_loss = self.d_loss_real + self.d_loss_fake
        

    def train(self, epochs=2, batch_size=245, learning_rate=0.0002, beta1=0.5, pre_trained_model=None):
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
        
        with tf.variable_scope('optim', reuse=tf.AUTO_REUSE):
            # using Adam optimizer as specified in the project paper
            d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.d_loss) #generator optimizer
            g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.g_loss) # discriminator optimizer
      
        counter = 0
        cur_model_name = 'DCGAN_{}'.format(int(time.time()))
        
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), self.sess.graph)
        saver = tf.train.Saver()
        
        start_time = time.time()
        
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                raise ValueError("Load model Failed!")
                
                
        # run
        #with tf.Session() as sess:
           # sess.run(init)
            
        #generate batches based on which dataset is being used
        #if self.data_set == '': --> uncomment if needed
        batches, iters = get_batches(batch_size, self.dataset)
                
        # getting number of training iterations
        #iters = 1000
        print('number of batches for training: {}'.format(iters))

        for epc in range(epochs):
            print("epoch {} ".format(epc+1))
            for i in range(iters):
                counter += 1
                #get batch
                batch_images = next(batches)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                #_, g_loss_value, d_loss_value = sess.run([train_op, losses[self.g], losses[self.d]])

                # ---> Run g_optim twice to make sure that d_loss does not go to zero?
                # Update D network
                _, D_loss_curr = self.sess.run([d_optim, self.d_loss], feed_dict={ self.inputs: batch_images, self.z: batch_z })

                # Update G network
                _, G_loss_curr = self.sess.run([g_optim, self.g_loss], feed_dict={ self.z: batch_z })

                # ---> Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, G_loss_curr = self.sess.run([g_optim, self.g_loss], feed_dict={ self.z: batch_z })
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})
                
                if counter % 10 == 1:
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epc+1, epochs, i, iters,
                        time.time() - start_time, errD_fake+errD_real, errG))
                
                if counter % 100 == 1:
                    # do validation
                    sample_in = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)
                    sample_input = self.sampler(sample_in)
                    samples = self.sess.run([sample_input])
                    #print(samples)
                    #rescale generated image
                    samples = (samples[0] + 1) * 255 / 2
                    
                    
                    #show generated image
                    img=samples[0,:,:,:]
                    #print(img)
                    img = img.astype(int)
                    plt.imshow(img)
                    plt.show()

                    plt.savefig('gen_train_img/img_{}'.format(counter+epc))
               
                    # Save checkpoint
                    saver.save(self.sess, 'model/{}'.format(cur_model_name))
        print("Training complete. Model named {}.".format(cur_model_name))
    
    def sampler(self, z, y=None):
        # This function creates sample images for validation in the training function
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
            #scope.reuse_variables()
            
            #reshape from inputs            
            reshape_out = tf.layers.dense(z, self.g_dim[0] * self.s_size * self.s_size)
            reshape_out = tf.reshape(reshape_out, [-1, self.s_size, self.s_size, self.g_dim[0]])
            reshape_out = tf.nn.relu(tf.layers.batch_normalization(reshape_out, training=False), name='g_reshape')
            
            # deconv layer 1
            conv_1 = tf.layers.conv2d_transpose(reshape_out, self.g_dim[1], [5, 5], strides=(2, 2), padding='SAME')
            conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=False), name='g_conv_1')
            
            # deconv layer 2
            conv_2 = tf.layers.conv2d_transpose(conv_1, self.g_dim[2], [5, 5], strides=(2, 2), padding='SAME')
            conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=False), name='g_conv_2')
            
            # deconv layer 3
            conv_3 = tf.layers.conv2d_transpose(conv_2, self.g_dim[3], [5, 5], strides=(2, 2), padding='SAME')
            conv_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=False), name='g_conv_3')
            
            # deconv layer 4
            conv_4 = tf.layers.conv2d_transpose(conv_3, self.g_dim[4], [5, 5], strides=(2, 2), padding='SAME')
            
            #output image
            return(tf.nn.tanh(conv_4))
            
        