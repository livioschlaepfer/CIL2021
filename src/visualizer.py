import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import numpy as np


def visualize_output(outputs, inputs, config, phase, labels=None):
    print("Started visualizer")

    if config.model_name == "bayesian_Unet":
        if phase == "train" or phase == "val":
            for (out, inp, lab) in zip(outputs, inputs, labels):
                binary = out.detach().clone()
                binary[binary>config.predict_threshold] = 1
                binary[binary<=config.predict_threshold] = 0
                
                lab = torch.tensor(lab, dtype=torch.float64)
                print(out.shape)
                input = transforms.ToPILImage(mode="RGB")(inp)
                label = transforms.ToPILImage(mode="L")(lab).convert("RGB")
                output = transforms.ToPILImage(mode="L")(out).convert("RGB")
                binary = transforms.ToPILImage(mode="L")(binary).convert("RGB")

                Image.fromarray(np.hstack((np.array(input), np.array(label), np.array(output), np.array(binary)))).show()
                break

        if phase == "test":
            for (out, inp) in zip(outputs, inputs):
                binary = out.detach().clone()
                binary[binary>config.predict_threshold] = 1
                binary[binary<=config.predict_threshold] = 0
                        
                input = transforms.ToPILImage(mode="RGB")(inp)
                output = transforms.ToPILImage(mode="L")(out).convert("RGB")
                binary = transforms.ToPILImage(mode="L")(binary).convert("RGB")

                Image.fromarray(np.hstack((np.array(input), np.array(output), np.array(binary)))).show()

    if config.model_name == "deeplabv3":
        for idx, output in enumerate(outputs):
            if idx == 0:
                # convert output to binary encoding
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
