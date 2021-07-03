from __future__ import print_function
from __future__ import division

import numpy as np

import torch
from torch.nn.functional import log_softmax
import time
import copy
from tqdm import tqdm

from src.visualizer import visualize_output
from src.criterion.dice_loss import DiceLoss

def train_model(runner, dataloaders, optimizer, scheduler, device, config, num_epochs=25):
    since = time.time()
    vis_time = time.time()


    val_acc_history = []

    best_model_wts = copy.deepcopy(runner.model.state_dict())
    best_acc = 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                runner.model.train()  # Set model to training mode
            else:
                runner.model.eval()   # Set model to evaluate mode
                val_loss = 0          # init val loss

            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch") as tqdm_dataloader:
                for inputs, labels in tqdm_dataloader:
                    tqdm_dataloader.set_description(f"Epoch {epoch}")

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss

                        outputs = runner.forward(inputs)

                        # Visualize output
                        if config.visualize_model_output and (time.time()-vis_time>config.visualize_time):
                            visualize_output(outputs, inputs, labels, config=config)
                            vis_time=time.time()
                        

                        loss = runner.criterion(outputs.float(), labels.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            # print("=========================== updated weights")

                        if phase == 'val':
                            val_loss = val_loss + loss.item()
                        
                        # batch statistics
                        dice = DiceLoss()
                        tqdm_dataloader.set_postfix(loss = loss.item(), accuracy = 100. * dice(outputs.float(), labels.float()).item())

            # deep copy the model
            if phase == 'val' and loss < best_acc:
                print("Found new best weight, epoch", epoch)
                config.best_epoch = epoch
                best_acc = loss
                best_model_wts = copy.deepcopy(runner.model.state_dict())
            if phase == 'val':
                val_acc_history.append(loss)
                print("=" * 20 , "validation loss:", val_loss)

        # update lr at the end of epoch
        if config.lr.lr_policy == 'plateau':
                scheduler.step(val_loss)
        else:
            scheduler.step()
        print("new lr: ", optimizer.param_groups[0]['lr'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    runner.model.load_state_dict(best_model_wts)
    return runner, val_acc_history