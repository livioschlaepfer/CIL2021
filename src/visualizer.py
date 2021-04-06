import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import numpy as np


def visualize_output(outputs, inputs, labels, config):
    print("Started visualizer")

    for (out, inp, lab) in zip(outputs, inputs, labels):
        binary = out.detach().clone()
        binary[binary>config.predict_threshold] = 1
        binary[binary<=config.predict_threshold] = 0
        
        lab = torch.tensor(lab, dtype=torch.float64)
        
        input = transforms.ToPILImage(mode="RGB")(inp)
        label = transforms.ToPILImage(mode="L")(lab).convert("RGB")
        output = transforms.ToPILImage(mode="L")(out).convert("RGB")
        binary = transforms.ToPILImage(mode="L")(binary).convert("RGB")

        Image.fromarray(np.hstack((np.array(input), np.array(label), np.array(output), np.array(binary)))).show()