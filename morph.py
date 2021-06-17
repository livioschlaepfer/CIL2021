import cv2
from skimage.morphology import erosion, dilation, rectangle, octagon, closing, disk, skeletonize
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    plt.show()


img_0 = cv2.imread("/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/predictions/no_blur_test/test_7.png", cv2.IMREAD_GRAYSCALE)
_, img_0 = cv2.threshold(img_0, 127, 1, cv2.THRESH_BINARY)

selem = disk(20)
print(np.array(img_0).shape)

eroded = erosion(np.array(img_0), selem)
dilated = dilation(eroded, selem)
closed = closing(img_0, selem)
skelet = skeletonize(closed)
plot_comparison(img_0, skelet, 'modified')

#print(img_0)

print("done")
