import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from imgaug import augmenters as iaa
from IPython.display import display
import sys
from util.visualizer import save_images
from util.util import save_image
import glob
import os

def crf_postprocessing(paths):
    """ take the output of the model and apply crf postprocessing """

    # convert the input image to an rgb
    input_rgb = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
    
    # coerce mask in range 0 to 255
    mask = cv2.imread(paths[1], 0)
    # create "negative" of mask by bitwise operation
    not_mask = cv2.bitwise_not(mask)

    # expand dimensions
    mask = np.expand_dims(mask, axis=2)
    not_mask = np.expand_dims(not_mask, axis=2)

    # stacka and convert to softmax values
    softmax = np.concatenate([not_mask, mask], axis=2)/255
    print(softmax.shape)
    feat_first = np.squeeze(softmax) #.transpose((2, 0, 1)).reshape((2,-1))

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

img_paths_mask = glob.glob('/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/Livio_pred'+'/*.png')
img_paths = glob.glob('/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images/test_images'+'/*.png')

for paths in zip(img_paths, img_paths_mask):
    crf_mask = crf_postprocessing(paths)
    save_image(crf_mask,'/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/Livio_pred/after_crf/'+os.path.split(paths[1])[1]+'.png')



