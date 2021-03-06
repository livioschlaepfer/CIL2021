def paths_setter(username, pretrain=False, mixed_train=False):
        if username == "Mathiass" and pretrain==False and mixed_train==False:
                path_dict = {
                        'train_mask_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\training\training\groundtruth",
                        'train_image_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\training\training\images",
                        'test_image_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\test_images\test_images",
                        'test_output_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\checkpoints",
                        'model_store': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\modelstore",
                }
        
        elif username == "Mathiass" and pretrain==True:
                path_dict = {
                        'train_mask_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\Massachusetts Road Dataset\Masks",
                        'train_image_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\Massachusetts Road Dataset\Images",
                        'test_image_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\test_images\test_images",
                        'test_output_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\checkpoints",
                        'model_store': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\modelstore",
                }

        elif username == "Mathiass" and mixed_train==True:
                path_dict = {
                        'train_mask_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\training\training\groundtruth",
                        'train_image_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\training\training\images",
                        'test_image_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\test_images\test_images",
                        'test_output_dir': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\checkpoints",
                        'model_store': r"C:\Users\Mathiass\Documents\Python Scripts\FS2021_CIL_Code\project3_road_segmentation\modelstore",
                }
        if username == "svkohler" and pretrain==False and mixed_train==False:
            path_dict = {
                    'train_mask_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth',
                    'train_image_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images',
                    'test_image_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images/test_images',
                    'test_output_dir': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/checkpoints',
                    'model_store': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/checkpoints',
                    'massa_images': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/input",
                    'massa_masks': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/output',
                    'massa_images_aug': "/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/input_aug",
                    'massa_masks_aug': '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/massa/output_aug'
            }
        
        elif username == "svkohler" and pretrain==True:
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

        elif username == "svkohler" and mixed_train==True:
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

        elif username == 'livioschlapfer':
            path_dict = {
                    'train_mask_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth',
                    'train_mask_dir_aug_input': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth', 
                    'train_mask_dir_aug_output': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth_aug', 
                    'train_image_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images',
                    'train_image_dir_aug_input': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images',
                    'train_image_dir_aug_output': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images_aug',
                    'test_image_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/test_images/test_images',
                    'test_output_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/test_images/test_output',
                    'model_store': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/model_store',
            }
        
        elif username == 'livios':
            path_dict = {
                    'train_mask_dir': '/cluster/home/livios/data/training/training/groundtruth',
                    'train_mask_dir_aug_input': '/cluster/home/livios/data/training/training/groundtruth', 
                    'train_mask_dir_aug_output': '/cluster/home/livios/data/training/training//groundtruth_aug', 
                    'train_image_dir': '/cluster/home/livios/data/training/training/images',
                    'train_image_dir_aug_input': '/cluster/home/livios/data/training/training/images',
                    'train_image_dir_aug_output': '/cluster/home/livios/data/training/training/images_aug',
                    'test_image_dir': '/cluster/home/livios/data/test_images/test_images',
                    'test_output_dir': '/cluster/home/livios/data/test_images/test_output',
                    'model_store:': '/cluster/home/livios/data/model_store'
            }

        return path_dict