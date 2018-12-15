import numpy as np
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
from time import gmtime, strftime


pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)



def crop(image, cropx = 64, cropy = 64):
    '''
    crops an image
    '''
    y, x, _ = image.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return image[ starty : starty + cropy, startx : startx + cropx, :]

def block_mask(data,block_size):
############################################################################################
#This function applies a mask to the input dataset that removes a block from the center of 
#the images by setting those values to 0.
#
#Input Arguments:
#data:N x M x M x 3 numpy array of sqaure RGB images
#mask_size: int value for the size of the block to be cropped out of the datat set. 
#i.e (mask_size = 4 outputs the data with a 4x4 block of zeros in the center). mask_size must 
#not be larger than the dimensions of the data
#
#Output:
#masked_data: N x M x M x 3  numpy array with central block set to 0
#mask: set of masks
############################################################################################
    #return error message if block_size is too large
    if block_size >= data.shape[1]:
        print('error, block_size is larger than data')
        return 0
    else:
        #get shape of data
        M = 64
        N = data.shape[0]
        #create mask of ones of the same size
        mask = np.ones((1,M,M,3))
        #find starting index value for the center block
        start_point = np.ceil(M/2) - (np.ceil(block_size/2) - 1) - 1
        #set all values within center block of the mask to 0
        for k in range(3):
            for i in range(block_size):
                for j in range(block_size):
                    mask[0, int(start_point+i) ,int(start_point+j),k] = 0
        #multiply mask with all entries in data
        masked_data = np.zeros(N,M,M,3)
        for i in range(N):
            masked_data[i,:,:,:] = np.multiply(data[i,:,:,:],mask)
            
        return masked_data, mask



#def random_pattern_mask(data,percent):
############################################################################################
#This function applies a mask the input dataset that randomly removes a pattern of a set
#percentage from the input. a value of 25% is used within https://arxiv.org/abs/1607.07539
############################################################################################


def random_mask(data,percent):
#############################################################################################
#This function applies a mask to the input dataset that randomly removes a set percent of 
#the input. a value of 80% is used within https://arxiv.org/abs/1607.07539
#
#Input Arguments:
#data: N x M x 3  numpy array of sqaure RGB images
#percent: percentage of data to be randomly removed
#
#Output:
#masked_data: N x M x M x 3 numpy array with random values set to 0 based on set percentage
#mask: set of masks
############################################################################################

    import numpy as np

    #get shape of data set
    M = data.shape[1]
    N = data.shape[0]
    #create masked_data array
    masked_data = np.zeros((N,M,M,3))
    mask = np.zeros((N,M,M,3))
    for i in range(N):
        next_mask = np.random.binomial(1,1-percent,(M,M,1))
        mask[i,:,:,:] = np.tile(next_mask,3)
        masked_data[i,:,:] = np.multiply(data[i,:,:,:],mask[i,:,:,:])
    return masked_data, mask

def half_missing_mask(data):
############################################################################################
#This function randomly removes half of each input image either horizontally or vertically
#
#Input Arguments:
#data: Nx M x M x 3  numpy array of sqaure RGB images
#Output:
#masked_data: Nx M x M x 3 numpy array with half of each image blacked out either vertically
#or horizontally
#masks: set of masks
############################################################################################
    import numpy as np
    
    #get shape of data
    M = data.shape[1]
    N = data.shape[0]
    
    masked_data = np.zeros((N,M,M,3))
    mask = np.zeros((N,M,M,3))
    
    for i in range(N):
        if np.random.rand(1) > 0.5:
            next_mask = np.ones((1,M,M,3))
            next_mask[:,0:M , 0: int(np.ceil(M/2)),0:3] = 0
            mask[i,:,:,:] = next_mask
            masked_data[i,:,:,:] = np.multiply(data[i,:,:,:], mask[i,:,:,:])
        else:
            next_mask = np.ones((1,M,M,3))
            next_mask[:,0: int(np.ceil(M/2)), 0: M ,0:3] = 0
            mask[i,:,:,:] = next_mask
            masked_data[i,:,:,:] = np.multiply(data[i,:,:,:] , mask[i,:,:,:])
    return masked_data, mask

