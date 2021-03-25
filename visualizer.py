import torch
from PIL import Image
import matplotlib.pyplot as plt


def visualize_output(output, config):
    print("Started visualizer")

    input_image = Image.open(config.image_dir + '/satImage_001.png')

    for image in output:
        output_predictions = image.argmax(0)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(2)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        #plt.imshow(r)
        r.show()