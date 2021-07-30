# CIL Project on Road Segmentation

Welcome!

This is the Git Repository of Livio Schläpfer, Mathias Rouss and Sven Kohler containing the code to reproduce our experimental results for the Road Segmentation Challenge which was completed as part of the Computational Intelligence Lab 2021.

## Directory Structure
```
├── configs
│   ├── baseline_deeplab.yaml
│   ├── ...
│   └── custom.yaml
├── src
│   ├── criterion
│   │   ├── focal_loss.py
│   │   ├── dice_loss.py
│   │   └── ...
│   ├── models
│   │   ├── deeplabv3.py
│   │   └── ...
│   ├── trainer.py
│   ├── tester.py
│   └── ...
├── runner_training.py
├── runner_test.py
├── ...
└── mask_to_submission.py
```

Some notes:
1. Configurations for baselines are directly provided, while the remaining configurations of experiments are provided in our [Polybox Model Store](https://polybox.ethz.ch/index.php/s/qtn4FY23P8lj4xG?path=%2FExperiments). 
2. Directories `src/cirterion` and `src/models` contain implementations of different loss functions and model architectures used within our experiments. The remaining files in the `src` directory are helpers to the runner files.
4. Files starting with `runner_...` are used to do training / testing runs, see section [Reproducing Scores](#reproducing-scores).

## Getting started
**1) Requirements**

Preferably, create your own conda environment before following the step below:

```
pip3 install -r requirements.txt
```

**2) Download Data**

Please download the CIL Road Segmentation Dataset ([link](https://www.kaggle.com/c/cil-road-segmentation-2021/data)) and the used subset of the Massachusetts Road Dataset ([link](https://polybox.ethz.ch/index.php/s/p682LOyrCIHegpW)). 

**3) Setup paths**

Next, update the path variables under `src/paths.py` as described below:

```
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
```

**3) Polybox Model Store**

In our [Polybox Model Store](https://polybox.ethz.ch/index.php/s/qtn4FY23P8lj4xG?path=%2FExperiments) we provide configurations, predicted segmentation masks, submission files, and model weights of conducted experiments.

To skip training and directly start with predictions, download the desired experiment folder from the Polybox Model Store and place the folder within your local model store.


## Reproducing scores

The following steps will allow you to reproduce the presented scores. Please note that the scores presented in the report consist of mean and standard deviation calculated over 3 seeds. Thus, the below introduced procedures must be repeated 3 times with a different seed. Average / standard deviation must be computed manually based on the obtained submission scores.

To skip training and directly start with predictions, download the desired experiment folder from the Polybox Model Store and place the folder within your local model store.  

**Reproducing Baselines - Training**

Update the value of the config flag with the desired baseline and e.g. run:

```
python3 runner_training.py -config baseline_deeplab.yaml
```

**Reproducing Baselines - Prediction, Submission Mask**

Update the value of the config flag with the desired baseline and e.g. run:

```
python3 runner_test.py -config baseline_deeplab.yaml
python3 mask_to_submission.py -config baseline_deeplab.yaml
```

**Reproducing Experiments - Training**

Update `custom.yaml` based on the configuration details of the desired exeriment. The configuration details are available in the PolyBox Model Store. Then run:
```
python3 runner_training.py -config custom.yaml
```

**Reproducing Experiments - Prediction, Submission Mask**

Update `custom.yaml` based on the configuration details of the desired exeriment. The configuration details are available in the PolyBox Model Store. Then run:
```
python3 runner_test.py -config custom.yaml
python3 mask_to_submission.py -config custom.yaml
```

**Reproducing Final Kaggle Submission**

The final Kaggle submission score was obtained by averaging over the outputs of the experiments for [deeplab_bce](https://polybox.ethz.ch/index.php/s/ZYqomRNR52tY3Jl), [deeplab_focal](https://polybox.ethz.ch/index.php/s/bTY4hLZqSxMgOSl), [deeplab_dice](https://polybox.ethz.ch/index.php/s/qhxmlNXiAC3FOTY) over all 3 seeds. Either reproduce the submission scores from scratch according to steps introduced above or download the experiments folders from our [Polybox Model Store](https://polybox.ethz.ch/index.php/s/qtn4FY23P8lj4xG?path=%2FExperiments).

After producing the prediction segmentation mask for the mentioned experiments, average the outputs and obtain the submission file by running:
```
python3 runner_majority_voting.py majority_all_maj --models "deeplab_trainaug_testaug,deeplab_trainaug_testaug,deeplab_trainaug_testaug,deeplab_focal,deeplab_focal,deeplab_focal,deeplab_dice,deeplab_dice,deeplab_dice" --model_seeds "1,2,3,1,2,3,1,2,3"
python3 mask_to_submission.py -config custom.yaml
```

## Configuration files

To reproduce the scores presented in our report the interested user will mainly interact with the configuration files under `configs/`. Below you find a detailed list of available settings,flags, and variables:

- model_name: str, refers to the model/baseline to be used. Choices are ["unet_baseline", "fcnres50_baseline", "deeplabv3"]
- checkpoint_name: str, name of the current experiment. Creates folder in path_dict["model_store"] and saves corresponding data (weights, predictions etc.)
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

