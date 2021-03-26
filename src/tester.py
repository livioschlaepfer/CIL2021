from __future__ import print_function
from __future__ import division

import os
import glob
import torch
from PIL import Image

from src.visualizer import visualize_output

def test_model(model, dataloaders, device, config):

    # Get sample image shape
    input_image = Image.open(config.paths.train_image_dir + '/satImage_001.png')

    # Create output folder if not existing
    if not os.path.exists(config.paths.test_output_dir):
        os.makedirs(config.paths.test_output_dir)

    # Set model to evaluate mode
    model.eval()
    phase = 'test'

    print("Start creating outputs")

    with torch.no_grad():
        # Iterate over data.
        for inputs, path in dataloaders[phase]:
            inputs = inputs.to(device)
        
            # Get outputs
            outputs = model(inputs)['out']

            # Visualize output #TODO: Only for testing
            if config.visualize_model_output:
                visualize_output(outputs, config=config)

            # Convert output to .png and store
            for output in outputs:
                output_predictions = output.argmax(0)

                palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                colors = torch.as_tensor([i for i in range(2)])[:, None] * palette
                colors = (colors % 255).numpy().astype("uint8")

                # Plot the semantic segmentation predictions per class
                r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
                r.putpalette(colors)

                # Store output
                r.save(config.paths.test_output_dir + "/" + path[0].split("/")[-1])
                print("Stored output for", path[0].split("/")[-1])

