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

    with torch.no_grad():
        # Iterate over data.
        for inputs, path in dataloaders[phase]:
            inputs = inputs.to(device)
        
            # Get outputs
            outputs = runner.forward(inputs, path)

            # Visualize output #TODO: Only for testing
            if config.visualize_model_output:
                visualize_output(outputs, config=config)

            # Convert output to .png and store
            runner.convert_to_png(outputs)
                

