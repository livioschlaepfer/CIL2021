"""This module contains simple helper functions """
from __future__ import print_function
from numpy.lib.type_check import imag
import torch
import numpy as np
from PIL import Image
import os
# crf postprocessing
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from imgaug import augmenters as iaa
from IPython.display import display
import sys



def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        #print("before sclaling, change type", image_numpy.astype(imtype))
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        #     print(image_numpy.shape)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image  
    image_numpy = np.squeeze(image_numpy)
    #print(image_numpy.shape)
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0, resize=None):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy, "L")
    h, w = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)

    if resize is not None:
        image_pil = image_pil.resize(resize, Image.BICUBIC)
    
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def crf_postprocessing(output, data):
    """ take the output of the model and apply crf postprocessing """

    # convert the input image to an rgb
    input_rgb = cv2.cvtColor(cv2.resize(cv2.imread(data["A_paths"][0]), (512,512)), cv2.COLOR_BGR2RGB)
    
    # coerce mask in range 0 to 255
    if isinstance(output, np.ndarray):
        mask = np.squeeze(output*255).astype(np.uint8)
    if isinstance(output, torch.Tensor):
        mask = np.squeeze(output.cpu().float().numpy()*255).astype(np.uint8)
    # create "negative" of mask by bitwise operation
    not_mask = cv2.bitwise_not(mask)

    # expand dimensions
    mask = np.expand_dims(mask, axis=2)
    not_mask = np.expand_dims(not_mask, axis=2)

    # stacka and convert to softmax values
    softmax = np.concatenate([not_mask, mask], axis=2)/255
    feat_first = np.squeeze(softmax).transpose((2, 0, 1)).reshape((2,-1))

    # get unary potentials 
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    # initialize the CRF object
    d = dcrf.DenseCRF2D(input_rgb.shape[0], input_rgb.shape[1], 2)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=input_rgb.copy(order='C'),
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # approximate inference
    Q = d.inference(5)

    # extract argmax
    res = np.argmax(Q, axis=0).reshape((input_rgb.shape[0], input_rgb.shape[1]))

    crf_mask = np.array(res*255, dtype=np.uint8)

    return crf_mask
