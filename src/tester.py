from __future__ import print_function
from __future__ import division

import os
import glob
import numpy as np
import torch
from PIL import Image
import time

from src.visualizer import visualize_output
from src.transforms_test import transform_test, transform_test_aggregate, transform_test_back

def test_model(runner, dataloaders_test, dataloaders_train, device, optimizer, config):   
    since = time.time()
    vis_time = time.time()

    # Create output folder if not existing
    if not os.path.exists(config.paths.test_output_dir):
        os.makedirs(config.paths.test_output_dir)

    print("Start creating outputs")

    # Apply pseudo labeling if configured
    if config.pseudo_labeling:

        phase = 'test'

        # Iterate over data.
        for inputs, path in dataloaders_test[phase]:

            inputs = inputs.to(device)

            # 1) Get pseudo label for test batch
            runner.model.eval()
            with torch.no_grad():
                print("shape inputs 1", inputs.shape)
                pseudo_labels = runner.forward(inputs)

            # Visualize output #TODO: Only for testing
            if config.visualize_model_output:
                visualize_output(pseudo_labels, inputs, config=config)
                vis_time=time.time()

            # 2) Update model with random batch from training data
            runner.model.train() 
            phase = 'train'

            batch_inputs, batch_labels = next(iter(dataloaders_train['train']))
            batch_inputs = batch_inputs.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = runner.forward(batch_inputs)
                
                loss1 = runner.criterion(outputs.float(), batch_labels.float())

                print("loss 1", loss1.shape)

            # 3) Evaluate on same test batch 

            with torch.set_grad_enabled(phase == 'train'):
                print("shape inputs 2", inputs.shape)
                outputs = runner.forward(inputs)
                
                loss2 = runner.criterion(outputs.float(), pseudo_labels.float())

                print("loss 2", loss2.shape)

                loss = 0.7 * loss1 + 0.3 * loss2

                loss.backward()
                optimizer.step()

            # Visualize output #TODO: Only for testing
            if config.visualize_model_output and (time.time()-vis_time>config.visualize_time):
                visualize_output(outputs, inputs, config=config)
                vis_time=time.time()
    

    # Set model to evaluate mode
    runner.model.eval()
    phase = 'test'

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
                # if config.visualize_model_output and (time.time()-vis_time>config.visualize_time):
                visualize_output(output, input, config=config)
                vis_time=time.time()

                # Convert output to .png and store
                png = runner.convert_to_png(output[0])

                # Store output
                png.save(config.paths.test_output_dir + "/" + os.path.split(paths[index])[1])
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
                png.save(config.paths.test_output_dir + "/" + os.path.split(paths[index])[1])
                print("Stored output for", os.path.split(paths[index])[1])

            

