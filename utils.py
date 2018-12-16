import numpy as np
import math
import random
import scipy.misc
from time import gmtime, strftime

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
        masked_data = np.zeros((N,M,M,3))
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

def save_image(image, path):
    image = (image + 1) / 2
    return scipy.misc.imsave(path, (255 * image).astype(np.uint8))

