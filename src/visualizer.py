import torch
from PIL import Image
import matplotlib.pyplot as plt


def visualize_output(output, config):
    print("Started visualizer")

    for image in output:
        output_predictions = image
        binary = output_predictions
        binary[binary>config.predict_threshold] = 1
        binary[binary<=config.predict_threshold] = 0

        two_img = vstack(binary.numpy(), output_predictions.numpy())

        # plot the semantic segmentation predictions per class
        r = Image.fromarray(output_predictions.cpu())

        #plt.imshow(r)
        r.show()