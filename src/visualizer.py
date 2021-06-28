import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualize_output(outputs, inputs = None, labels = None, config=None):
    #print("Started visualizer")

    for idx, output in enumerate(outputs):
        if idx == 0:
            # convert output to binary encoding
            binary = output.argmin(0)
            binary = torch.tensor(binary, dtype=torch.float64)
            binary = transforms.ToPILImage(mode="L")(binary).convert("RGB")

            if inputs is not None and labels is not None:
                input = transforms.ToPILImage(mode="RGB")(inputs[idx])
                label = transforms.ToPILImage(mode="LA")(labels[idx]).convert("RGB")
                
                # plot the semantic segmentation predictions per class
                Image.fromarray(np.hstack((np.array(input), np.array(label), np.array(binary)))).show()

            else:
                input = transforms.ToPILImage(mode="RGB")(inputs[idx])
                # plot the semantic segmentation predictions per class
                Image.fromarray(np.hstack((np.array(input), np.array(binary)))).show()
