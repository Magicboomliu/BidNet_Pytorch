from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")
from utils import utils
from utils.kitti_io import read_img,read_disp,read_kitti_image_step1,read_kitti_image_step2,read_kitti_step1,read_kitti_step2
from skimage import io, transform
import numpy as np
from PIL import Image

class ComplexKITTILoader(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='KITTI',
                 mode='train',
                 save_filename=False,
                 transform=None,
                 visible_list=[]):
        super(ComplexKITTILoader, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(384, 1280)
        self.scale_size =(384,1280)
        self.original_size =(375,1280)
        self.visible_list = visible_list
        

        sceneflow_finalpass_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'val': 'filenames/KITTI_2012_val.txt'
        }

        kitti_2015_dict = {
            'train': self.train_datalist,
            'test': self.test_datalist
        }

        kitti_mix_dict = {
            'train': self.train_datalist,
            'test': self.test_datalist
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

        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]
            
            left_disp_pseudo = splits[3]
            right_disp_pseudo = splits[4]
            baseline = splits[5]
            focal_length = splits[6]
            beta = splits[7]
            airlight = splits[8]
            
            sample = dict()
            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            if 'clear_left' in self.visible_list:
                sample['clear_left'] = os.path.join(data_dir, left_img)
            if 'clear_right' in self.visible_list:
                sample['clear_right'] = os.path.join(data_dir, right_img)
            if 'gt_disp' in self.visible_list:
                sample['gt_disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None
            if "left_pseudo_disp" in self.visible_list:
                sample['left_pseudo_disp'] = os.path.join(data_dir,left_disp_pseudo)
            if "right_pseudo_disp" in self.visible_list:
                sample['right_pseudo_disp'] = os.path.join(data_dir,right_disp_pseudo)
            if 'baseline' in self.visible_list:
                sample['baseline'] = np.array(float(baseline))
            if 'focal_length' in self.visible_list:
                sample['focal_length'] = np.array(float(focal_length)) 
            if 'beta' in self.visible_list:
                sample['beta'] = np.array(float(beta))
            if 'airlight' in self.visible_list:
                sample['airlight'] = np.array(float(airlight))


            self.samples.append(sample)



    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        if self.mode=='train':
            if 'clear_left' in self.visible_list:
                sample['clear_left'] = read_img(sample_path['clear_left'])  # [H, W, 3]
            if 'clear_right' in self.visible_list:
                sample['clear_right'] = read_img(sample_path['clear_right'])
        
        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        
        if sample_path['gt_disp'] is not None:
            sample['gt_disp'] = read_disp(sample_path['gt_disp'], subset=subset)  # [H, W]
            w = sample['gt_disp'].shape[-1]
        
        if sample_path['left_pseudo_disp'] is not None:
            sample['left_pseudo_disp'] = read_disp(sample_path['left_pseudo_disp'], subset=subset)
        
        if sample_path["right_pseudo_disp"] is not None:
            sample["right_pseudo_disp"] = read_disp(sample_path["right_pseudo_disp"],subset=subset)
        
        if sample_path['focal_length'] is not None:
            sample['focal_length'] = sample_path['focal_length']
        if sample_path['baseline'] is not None:
            sample['baseline'] = sample_path['baseline']
        if sample_path['beta'] is not None:
            sample['beta'] = sample_path['beta']
        if sample_path['airlight'] is not None:
            sample['airlight'] = sample_path['airlight']



        if self.mode=='test' or self.mode=='val':
            # Image Crop Operation
            
            if 'clear_left' in self.visible_list:
                left_im = read_kitti_image_step1(sample_path['clear_left']) #[H,W,3]        
                w, h = left_im.size
                left_image = left_im.crop((w-1280, h-384, w, h))
                sample['clear_left'] = read_kitti_image_step2(left_image)
                w1,h1 = left_image.size
                
            if 'clear_right' in self.visible_list:
                right_im = read_kitti_image_step1(sample_path['clear_right']) 
                w, h = right_im.size
                right_image = right_im.crop((w-1280, h-384, w, h))
                sample['clear_right'] = read_kitti_image_step2(right_image)
                w2,h2 = right_image.size
                
            if 'gt_disp' in self.visible_list:
                # Disparity Crop Operation
                gt_disp = read_kitti_step1(sample_path['gt_disp'])
                w, h = gt_disp.size
                dataL = gt_disp.crop((w-1280, h-384, w, h))
                dataL = read_kitti_step2(dataL)
                sample['gt_disp']= dataL
            
            if 'left_pseudo_disp' in self.visible_list:
                left_pseudo_disp =read_kitti_step1(sample_path['left_pseudo_disp'])
                w,h = left_pseudo_disp.size
                dataL = left_pseudo_disp.crop((w-1280, h-384, w, h))
                dataL = read_kitti_step2(dataL)
                sample['left_pseudo_disp']= dataL
                
            
            if 'right_pseudo_disp' in self.visible_list:
                right_pseudo_disp =read_kitti_step1(sample_path['right_pseudo_disp'])
                w,h = right_pseudo_disp.size
                dataL = right_pseudo_disp.crop((w-1280, h-384, w, h))
                dataL = read_kitti_step2(dataL)
                sample['right_pseudo_disp']= dataL
            
            
            if 'gt_disp' in self.visible_list:
                w = sample['gt_disp'].shape[-1]

            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size