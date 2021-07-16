# CIL2021

Welcome!

This is the Git Repo of Livio Schläpfer, Mathias Rouss and Sven Kohler containing the code to reproduce our results for Project 3 ("Road Segmentation") which was completed within the scope of the Computational Intelligence Lab 2021.

## High profile explanation of the different files
- runner_training.py: Execute to train selected configuration
- runner_test.py: Execute to produce prediction for the selected configuration
- mask_to_submission.py: Execute to translate image prediction into uploadable Kaggle submission
- massa.py: Execute to prepare Massachusetts Road Dataset for pretraining and mixed training
- source folder (src):
    - criterion: folder which holds the implementations of the different loss functions
    - models: folder which holds the files to initialize the different models used in our experiments
    - dataset.py: creates datasets and data loaders
    - edge.py: canny edge detector
    - paths.py: holds the dictionaries with all the relevant paths
    - scheduler.py: different learning rate scheduler
    - seed.py: seeding function to ensure reproducability
    - tester.py: function which defines the whole testing process
    - trainer.py function which defines the whole training process
    - transform_test.py: defines transformations which are performed at test time
    - transforms.py: defines transformations which are performed at train time
    - visualizer.py: framework to visually keep track of training progress

## Data preparation
Before trying to reproduce our results, please arrange the pretraining, training and test data as instructed below.
First the user has to complete the missing folder paths in paths.py:
"""
if username == "insert OS-username here" and pretrain==False and mixed_train==False:
            path_dict = {
                    'train_mask_dir': 'please insert path to train masks',
                    'train_image_dir': 'please insert path to train images',
                    'test_image_dir': 'please insert path to test masks',
                    'test_output_dir': 'please insert path to test images',
                    'model_store': 'please insert path where you want information about your models to be stored',
                    'massa_images': 'please insert path to the Massachusetts Road Dataset masks',
                    'massa_masks': 'please insert path to the Massachusetts Road Dataset images',
                    'massa_images_aug': 'please insert path where you want th',
                    'massa_masks_aug': 'please insert path to train masks'
            }
        
        elif username == "insert OS-username here" and pretrain==True:
            path_dict = {
                    'train_mask_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/output_aug/final',
                    'train_image_dir': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/input_aug/final",
                    'test_image_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images/test_images',
                    'test_output_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images',
                    'model_store': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/checkpoints',
                    'massa_images': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/input",
                    'massa_masks': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/output',
                    'massa_images_aug': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/input_aug",
                    'massa_masks_aug': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/output_aug'
            }

        elif username == "insert OS-username here" and mixed_train==True:
                path_dict = {
                        'train_mask_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/mixed_train/groundtruth',
                        'train_image_dir': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/mixed_train/images",
                        'test_image_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images/test_images',
                        'test_output_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images',
                        'model_store': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/checkpoints',
                        'massa_images': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/input",
                        'massa_masks': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/output',
                        'massa_images_aug': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/input_aug",
                        'massa_masks_aug': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/output_aug'
                }
"""

paths file etc


## Configuration files

To reproduce our results which were presented in our report the interested user will mainly interact with configuration files stored in the "configs" folder.
Below is a detailed list of all the eligible settings/flags/variables which need to be specified in order to launch a run.



-baseline_fnc.yaml
-baseline_unet.yaml
-baseline_deeplab.yaml
-custom.yaml
The first 3 files conveniently reproduce 

