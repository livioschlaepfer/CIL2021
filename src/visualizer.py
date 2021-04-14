import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualize_output(inputs, labels, outputs, config):
    print("Started visualizer")

    for idx, output in enumerate(outputs):
        # convert output to binary encoding
        binary = output.argmin(0)
        binary = torch.tensor(binary, dtype=torch.float64)


        print("binary type", type(binary))
        print("label type", type(labels[idx]))

        input = transforms.ToPILImage(mode="RGB")(inputs[idx])
        label = transforms.ToPILImage(mode="LA")(labels[idx]).convert("RGB")
        binary = transforms.ToPILImage(mode="L")(binary).convert("RGB")

        print("inputs shape", input.size)
        print("labels shape", label.size)
        print("binary shape", binary.size)

        # plot the semantic segmentation predictions per class
        Image.fromarray(np.hstack((np.array(input), np.array(label), np.array(binary)))).show()