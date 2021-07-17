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
                    'model_store': 'please insert path where you want information about your models to be stored (model weights, config, predictions, submission)',
            }
        
        elif username == "insert OS-username here" and pretrain==True:
            path_dict = {
                    'train_mask_dir': 'please insert path to the Massachusetts Road Dataset train masks',
                    'train_image_dir': 'please insert path to the Massachusetts Road Dataset train images',
                    'test_image_dir': 'please insert path to test masks',
                    'test_output_dir': 'please insert path to test images',
                    'model_store': 'please insert path where you want information about your models to be stored (model weights, config, predictions, submission)',
            }

        elif username == "insert OS-username here" and mixed_train==True:
                path_dict = {
                        'train_mask_dir': 'please insert path to the mixed dataset train masks',
                        'train_image_dir': 'please insert path to the mixed dataset train images',
                        'test_image_dir': 'please insert path to test masks',
                        'test_output_dir': 'please insert path to test images',
                        'model_store': 'please insert path where you want information about your models to be stored (model weights, config, predictions, submission)',
                }
"""
The three case distinctions refer to whether the user intends to pretrain, mixed train or standard train.


## Configuration files

To reproduce our results which were presented in our report the interested user will mainly interact with configuration files stored in the "configs" folder.
Below is a detailed list of all the eligible settings/flags/variables which need to be specified in order to launch a run.

- model_name: str, refers to the model/baseline to be used. Choices are ["unet_baseline", "fcnres50_baseline", "deeplabv3"]
- checkpoint_name: str, name of the current experiment. Creates folder path_dict["model_store"] and saves corresponding data (weights, predictions etc.)
- pretrain: bool, if run is to pretrain
- mixed_train: bool, if model is trained on mixed dataset
- num_classes: int, indicate model the number of classes to segment
- seed_run: int, ordinal number stating the seed run
- seed: int, random seed used for current seed_run
- batchsize: int, number of images per batch, memory dependent
- visualize_model_output: bool, whether visualization of training progress is desired
- visualize_time: int, interval in which a training example is depicted
- num_epochs: int, number of epochs to train the model on
- use_train_val_split: bool, whether you want to exclude the validation set from training
- val_size: float, size of validation set
- continue_trainig: bool, flag load previously trained model and continue training
- continue_training_on_checkpoint: str, checkpoint name of model to pretrain on
- transforms:
        - apply_test_transforms: bool, flag to indicate test time transformation
        - apply_transforms: bool, flag to indicate train time transformation
        - crop_size: int, crop size of original image and mask 
        - flip_prob: float, probability that a training example is flipped horizontally/vertically
        - rot_prob: float, probability that a training example is rotated at a random angle
        - canny: bool, whether to add the output of the canny edge detector to the original images
        - canny_filter_size: int, variable controlling filtering neighborhood of the canny edge detector
        - canny_thresh: int, threshold of canny edge detector
- morph: 
        - apply: bool, flag indicating if morphological postprocessing is applied to the final prediction of the model
        - area_closing: bool, flag indicating if area closing is applied to the final prediction of the model
        - area_opening: bool, flag indicating if area opening is applied to the final prediction of the model
        - binary_closing: bool, flag indicating if binary closing is applied to the final prediction of the model
        - iter: int, number if binary closing iterations to conduct
- loss:
        - name: str, desired loss to be used during training. Choices are ["bce", "dice", "cl_dice", "focal"]
        - patched_loss: bool, ??? is this implemented?
        - iter: int, hyperparameter of cl_dice. Controls degree of skeletonization
        - alpha: float, hyperparameter of cl_dice. Controls weight of soft_dice loss compared to soft_cl_dice loss.
        - smooth: float, smoothing parameter of cl_dice. 
        - bce_weight: float, weight of Binary Cross Entropy Loss when training on compound loss.
        - only_foreground: bool, flag indicating whether to apply cl_dice only to the road map.
- lr:
        - init_lr: float, initial learning rate to use during training
        - lr_policy: learning rate policy to apply during training. Choices are [none", linear", "step", "plateau"]
- best_epoch: None, placeholder to save number of best epoch during training.

The authors provided the user with the preset configuration files to implement the three baselines and a custom configuration file where an arbitrary setting can be implemented.

- baseline_fnc.yaml
- baseline_unet.yaml
- baseline_deeplab.yaml
- custom.yaml

## Reproducibility

To ensure reproducibility we fixed the random seeds in all the stochastic elements of our code (pytorch, numpy, random).

We ran all our experiments using three different seeds (seed run 1: ,2,3) to gain insights in the robustness of our results.

