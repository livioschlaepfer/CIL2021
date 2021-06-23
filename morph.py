import cv2
from skimage.morphology import erosion, dilation, rectangle, octagon, closing, disk, skeletonize, area_closing, area_opening
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


img_0 = cv2.imread("/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/predictions/test_92.png", cv2.IMREAD_GRAYSCALE)
_, img_0 = cv2.threshold(img_0, 127, 1, cv2.THRESH_BINARY)

selem = rectangle(10, 2)
selem1 = rectangle(2, 10)
selem2 = disk(10)

eroded = erosion(np.array(img_0), selem)
closed = closing(img_0, selem)
skelet = skeletonize(closed)
area_open = area_opening(img_0, 1000)
dilated = dilation(area_open, selem)
inter = closing(area_open, selem)
closed = closing(inter, selem1)
plot_comparison(img_0, closed, 'modified')

#print(img_0)

print("done")
