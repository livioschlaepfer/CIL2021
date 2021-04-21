class paths_setter:
    def __init__(self, username):
        if username == "svenk":
            self.path_dict = {
                    'train_mask_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth/1',
                    'train_image_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images/1',
                    'test_image_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images/test_images',
                    'test_output_dir': 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images',
            }

        elif username == 'livioschlapfer':
            self.path_dict = {
                    'train_mask_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth_aug',
                    'train_mask_dir_aug': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/groundtruth_aug', 
                    'train_image_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images_aug',
                    'train_image_dir_aug': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/training/training/images_aug',
                    'test_image_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/test_images/test_images',
                    'test_output_dir': '/Users/livioschlapfer/Documents/OneDrive/Daten Livio/Schule/ETH/Master/2. Semester/Computational Intelligence Lab/3 - Project/2 - Data/cil-road-segmentation-2021/test_images/test_output',
            }
