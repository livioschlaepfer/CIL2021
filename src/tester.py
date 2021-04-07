from __future__ import print_function
from __future__ import division

import os
import glob
import torch
from PIL import Image

from src.visualizer import visualize_output

def test_model(runner, dataloaders, device, config):   

    # Create output folder if not existing
    if not os.path.exists(config.paths.test_output_dir):
        os.makedirs(config.paths.test_output_dir)

    # Set model to evaluate mode
    runner.model.eval()
    phase = 'test'

    print("Start creating outputs")

    # Iterate over data.
    for inputs, path in dataloaders[phase]:

        inputs = inputs.to(device)
    
        # Get outputs
        with torch.no_grad():
            outputs = runner.forward(inputs)

        # Visualize output #TODO: Only for testing
        if config.visualize_model_output:
            visualize_output(outputs, inputs, config=config, phase=phase)

        for output in outputs:
            # Convert output to .png and store
            png = runner.convert_to_png(output)

            # Store output
            png.save(config.paths.test_output_dir + "/" + os.path.split(path[0])[1])
            print("Stored output for", os.path.split(path[0])[1])
                

