def paths_setter(username):
        if username == "svenk":
            path_dict = {
                    'train_mask_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth_aug',
                    'train_mask_dir_aug_input': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth',
                    'train_mask_dir_aug_output': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth_aug',
                    'train_image_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images_aug',
                    'train_image_dir_aug_input': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images',
                    'train_image_dir_aug_output': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images_aug',
                    'test_image_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images/test_images',
                    'test_output_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images',
            }

        elif username == 'livioschlapfer':
            path_dict = {
                    'train_mask_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth_aug',
                    'train_mask_dir_aug_input': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth', 
                    'train_mask_dir_aug_output': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth_aug', 
                    'train_image_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images_aug',
                    'train_image_dir_aug_input': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images',
                    'train_image_dir_aug_output': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images_aug',
                    'test_image_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/test_images/test_images',
                    'test_output_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/test_images/test_output',
            }
        
        elif username == 'livios':
            path_dict = {
                    'train_mask_dir': '/cluster/home/livios/data/training/training/groundtruth_aug',
                    'train_mask_dir_aug_input': '/cluster/home/livios/data/training/training/groundtruth', 
                    'train_mask_dir_aug_output': '/cluster/home/livios/data/training/training//groundtruth_aug', 
                    'train_image_dir': '/cluster/home/livios/data/training/training/images_aug',
                    'train_image_dir_aug_input': '/cluster/home/livios/data/training/training/images',
                    'train_image_dir_aug_output': '/cluster/home/livios/data/training/training/images_aug',
                    'test_image_dir': '/cluster/home/livios/data/test_images/test_images',
                    'test_output_dir': '/cluster/home/livios/data/test_images/test_output',
            }

        return path_dict