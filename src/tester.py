from __future__ import print_function
from __future__ import division

import os
import torch
import time

from src.visualizer import visualize_output
from src.transforms_test import transform_test, transform_test_aggregate, transform_test_back

def test_model(runner, dataloaders_test, device, config, model=None):   
    since = time.time()
    vis_time = time.time()

    # Create output folder if not existing
    if not os.path.exists(config.paths.test_output_dir):
        os.makedirs(config.paths.test_output_dir)

    print("Start creating outputs")

    # Set model to evaluate mode
    runner.model.eval()
    phase = 'test'

    # Set output path
    output_path = config.paths.test_output_dir + "/" + model +'/predictions_seed_'+str(config.seed_run)+'/'
    print("Outputs are stored under", output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over data.
    for inputs, paths in dataloaders_test[phase]:

        if config.transforms.apply_test_transforms:

            for index, input in enumerate(inputs):
                # Apply test transforms
                input = transform_test(input)
                # Send to device
                input = input.to(device)
                # Get outputs
                with torch.no_grad():
                    outputs = runner.forward(input)
                # Aggregate from transforms
                outputs = transform_test_back(outputs)
                output = transform_test_aggregate(outputs)

                # Visualize output #TODO: Only for testing
                if config.visualize_model_output and (time.time()-vis_time>config.visualize_time):
                    visualize_output(output, input, config=config)
                    vis_time=time.time()

                # Convert output to .png and store
                png = runner.convert_to_png(output.squeeze())

                # Store output
                png.save(output_path + os.path.split(paths[index])[1])
                print("Stored output for", os.path.split(paths[index])[1])

        else: 
            inputs = inputs.to(device)
        
            # Get outputs
            with torch.no_grad():
                outputs = runner.forward(inputs)

            # Visualize output #TODO: Only for testing
            if config.visualize_model_output and (time.time()-vis_time>config.visualize_time):
                visualize_output(outputs, inputs, config=config)
                vis_time=time.time()

            for index in range(len(outputs)):
                # Convert output to .png and store
                png = runner.convert_to_png(outputs[index])

                # Store output
                png.save(output_path + os.path.split(paths[index])[1])
                print("Stored output for", os.path.split(paths[index])[1])

            

