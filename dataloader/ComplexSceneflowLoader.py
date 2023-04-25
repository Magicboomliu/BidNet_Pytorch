from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import sys
sys.path.append("..")
from utils.utils import read_text_lines
from utils.file_io import read_disp,read_img
from skimage import io, transform
import numpy as np

class ComplexStereoDataset(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 transform=None,
                 visible_list = []
                 ):
        super(ComplexStereoDataset, self).__init__()

        
        self.visible_list = visible_list
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(540, 960)
        self.scale_size =(576,960)
        

        sceneflow_finalpass_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/KITTI_mix.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = read_text_lines(data_filenames)

        for idx, line in enumerate(lines):
            splits = line.split()

            '''
            (0) clear left images. (1) clear right images.
            (2) haze left images.  (3) haze right images.
            (4) left disparity map. (5) right disparity map.
            (6) focal length.       (7) baseline
            (8) coefficients        (9) Airlight
            '''
            clear_left_image = splits[0]
            clear_right_image = splits[1]
            left_disparity_map = splits[2]
            right_disparity_map = splits[3]
            focal_length = float(splits[4])
            baseline = float(splits[5])
            scattering_coefficients = float(splits[6])
            airlight = float(splits[7])
            
            
            sample = dict()

            if self.save_filename:
                sample['left_name'] = clear_left_image.split('/', 1)[1]
            if 'clear_left_image' in self.visible_list:
                sample['clear_left_image'] = os.path.join(data_dir, clear_left_image)
            if 'clear_right_image' in self.visible_list:
                sample['clear_right_image'] = os.path.join(data_dir, clear_right_image)
            if 'left_disp' in self.visible_list:
                sample['left_disp'] = os.path.join(data_dir,left_disparity_map)
            if 'right_disp':
                sample['right_disp'] = os.path.join(data_dir,right_disparity_map)
            if 'focal_length' in self.visible_list:
                sample['focal_length'] = np.array(focal_length)
            if 'baseline' in self.visible_list:
                sample['baseline'] = np.array(baseline)
            if 'beta' in self.visible_list:
                sample['beta'] = np.array(scattering_coefficients)
            if 'airlight' in self.visible_list:
                sample['airlight'] = np.array(airlight)
            # Add these things into the list
            self.samples.append(sample)


    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        # Load the left image and right image both in clear scenes and the foggy scenes.
        if 'clear_left_image' in self.visible_list:
            sample['clear_left_image'] = read_img(sample_path['clear_left_image'])  # [H, W, 3]
        if "clear_right_image" in self.visible_list:
            sample['clear_right_image'] = read_img(sample_path['clear_right_image'])
        
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['left_disp'] is not None:
            sample['left_disp'] = read_disp(sample_path['left_disp'], subset=subset)  # [H, W]
        if sample_path['right_disp'] is not None:
            sample['right_disp'] = read_disp(sample_path['right_disp'], subset=subset)  # [H, W]
        if sample_path['focal_length'] is not None:
            sample['focal_length'] = sample_path['focal_length']
        if sample_path['baseline'] is not None:
            sample['baseline'] = sample_path['baseline']
        if sample_path['beta'] is not None:
            sample['beta'] = sample_path['beta']
        if sample_path['airlight'] is not None:
            sample['airlight'] = sample_path['airlight']
        
        if self.mode=='test' or self.mode=='val':
            # Re-Scale the Image.
            # process the clear left image and the clear right image.
            if 'clear_left_image' and 'clear_right_image' in self.visible_list:
                clear_img_left = transform.resize(sample['clear_left_image'], [576,960], preserve_range=True)
                clear_img_right = transform.resize(sample['clear_right_image'], [576,960], preserve_range=True)
                clear_img_left = clear_img_left.astype(np.float32)
                clear_img_right = clear_img_right.astype(np.float32)
                sample['clear_left_image'] = clear_img_left
                sample['clear_right_image'] = clear_img_right 
            


        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size