#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import glob
from box import Box
import yaml
import getpass
import argparse

from src.paths import paths_setter


foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    path_tail = os.path.split(image_filename)[1]
    img_number = int(re.search(r"\d+", path_tail).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':

    # parser to select desired
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
        default = 'custom',
        choices = ['custom', 'baseline_fcn', 'baseline_unet', 'baseline_deeplab'],
        help = 'Select on of the experiments described in our report or setup a custom config file'
    )
    parser.add_argument('--flag_majority', action='store_true', help="whether predictions come from a majority voting run")
    parser.add_argument('--majority_name', default=None, type=str, help="supply name of majority run")
    args = parser.parse_args()

    # load config
    try: 
        config = Box.from_yaml(filename="./configs/"+ args.config + ".yaml", Loader=yaml.FullLoader)
    except:
        raise OSError("Does not exist", args.config)

    print("Load config")

    # list all models to run tester

    # update paths based on user name
    username = getpass.getuser()
    config.paths = paths_setter(username=username)

    models = os.listdir(config.paths.model_store)

    if args.flag_majority:
        # check if dir exists, otherwise create
            if not os.path.exists(config.paths.model_store + "/" + args.majority_name +"/"+ 'submission/'):
                os.makedirs(config.paths.model_store + "/" + args.majority_name +"/"+ 'submission/')

            submission_filename = config.paths.model_store + "/" + args.majority_name +"/"+ 'submission/submission.csv'
            image_filenames = glob.glob(config.paths.model_store + "/" + args.majority_name +"/"+ 'predictions/*.png')
            
            print("Start masks to submission")

            masks_to_submission(submission_filename, *image_filenames)
    else:
        for model in models:
            # check if dir exists, otherwise create
            if not os.path.exists(config.paths.model_store + "/" + model +"/"+ 'submission_seed_' + str(config.seed_run)+'/'):
                os.makedirs(config.paths.model_store + "/" + model +"/"+ 'submission_seed_' + str(config.seed_run)+'/')

            submission_filename = config.paths.model_store + "/" + model +"/"+ 'submission_seed_' + str(config.seed_run)+'/' 'submission.csv'
            image_filenames = glob.glob(config.paths.model_store + "/" + model +"/"+ 'predictions_seed_' + str(config.seed_run) +"/*.png")

            print("Start masks to submission")

            masks_to_submission(submission_filename, *image_filenames)
