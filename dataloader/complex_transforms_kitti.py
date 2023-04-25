from __future__ import division
from genericpath import samefile
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
import cv2

# Compose.
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensorV2(object):
    """Convert numpy array to torch tensor"""
    def __call__(self, sample):
        left = np.transpose(sample['img_left'], (2, 0, 1))  # [3, H, W]
        sample['img_left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['img_right'], (2, 0, 1))
        sample['img_right'] = torch.from_numpy(right) / 255.

        # disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        if 'gt_disp_left' in sample.keys():
            disp = sample['gt_disp_left']  # [H, W]
            sample['gt_disp_left'] = torch.from_numpy(disp)
        if 'gt_disp_right' in sample.keys():
            disp = sample['gt_disp_right']  # [H, W]
            sample['gt_disp_right'] = torch.from_numpy(disp)
        return sample



# To Tensor
class ToTensor(object):
    """Convert numpy array to torch tensor"""
    def __call__(self, sample):
        # Clear Left Image
        if 'clear_left' in sample.keys():
            # print(sample['clear_left_image'].shape)
            clear_left_image = np.transpose(sample['clear_left'], (2, 0, 1))  # [3, H, W]
            sample['clear_left'] = torch.from_numpy(clear_left_image) / 255.
        # Clear Right Image
        if 'clear_right' in sample.keys():
            clear_right_image = np.transpose(sample['clear_right'], (2, 0, 1))
            sample['clear_right'] = torch.from_numpy(clear_right_image) / 255.
        if 'gt_disp' in sample.keys():
            gt_disp = sample['gt_disp']
            sample['gt_disp'] = torch.from_numpy(gt_disp)
        # left disparity
        if "left_pseudo_disp" in sample.keys():
            left_disp = sample['left_pseudo_disp']  # [H, W]
            sample['left_pseudo_disp'] = torch.from_numpy(left_disp)
        # right disparity
        if "right_pseudo_disp" in sample.keys():
            right_disp = sample['right_pseudo_disp']  # [H, W]
            sample['right_pseudo_disp'] = torch.from_numpy(right_disp)
        # focal length
        if "focal_length" in sample.keys():
            sample['focal_length'] = torch.from_numpy(sample['focal_length'])
        # baseline
        if 'baseline' in sample.keys():
            sample['baseline'] = torch.from_numpy(sample['baseline'])
        # beta
        if 'beta' in sample.keys():
            sample['beta'] = torch.from_numpy(sample['beta'])        
        # airlight
        if 'airlight' in sample.keys():
            sample['airlight'] = torch.from_numpy(sample['airlight']) 

        return sample

# Image Normalization 
class Normalize(object):
    """Normalize image, with type tensor"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        norm_keys = ['clear_left', 'clear_right']
        for key in norm_keys:
            if key in sample.keys():
            # Images have converted to tensor, with shape [C, H, W]
                for t, m, s in zip(sample[key], self.mean, self.std):
                    t.sub_(m).div_(s)
        return sample
    
    

# Random Crop
class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        if 'clear_left' in sample.keys():
            ori_height, ori_width = sample['clear_left'].shape[:2]
            
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            if 'clear_left' in sample.keys():
                sample['clear_left'] = np.lib.pad(sample['clear_left'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            
            if 'clear_right' in sample.keys():
                sample['clear_right'] = np.lib.pad(sample['clear_right'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            if 'gt_disp' in sample.keys():
                sample['gt_disp'] = np.lib.pad(sample['gt_disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)
        
            if 'left_pseudo_disp' in sample.keys():
                sample['left_pseudo_disp'] = np.lib.pad(sample['left_pseudo_disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)
            if 'right_pseudo_disp' in sample.keys():
                sample['right_pseudo_disp'] = np.lib.pad(sample['right_pseudo_disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)
        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:
                self.offset_x = np.random.randint(ori_width - self.img_width + 1)
                start_height = 0
                assert ori_height - start_height >= self.img_height
                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2
            
            # Crop The Clear Images.
            if 'clear_left' in sample.keys():
                sample['clear_left'] = self.crop_img(sample['clear_left'])
            if 'clear_right' in sample.keys():
                sample['clear_right'] = self.crop_img(sample['clear_right'])
            # Crop the Hazing Images.
            if 'left_pseudo_disp' in sample.keys():
                sample['left_pseudo_disp'] = self.crop_img(sample['left_pseudo_disp'])
            if 'right_pseudo_disp' in sample.keys():
                sample['right_pseudo_disp'] = self.crop_img(sample['right_pseudo_disp'])
            if 'gt_disp' in sample.keys():
                sample['gt_disp'] = self.crop_img(sample['gt_disp'])
                
        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]


# RandomVeticalFilp Operation.
class RandomVerticalFlip(object):
    """Randomly vertically filps"""
    def __call__(self, sample):
        if np.random.random() < 0.09:
            # Random Vertical Filped
            if 'clear_left' in sample.keys():
                sample['clear_left'] = np.copy(np.flipud(sample['clear_left']))
            if 'clear_right' in sample.keys():
                sample['clear_right'] = np.copy(np.flipud(sample['clear_right']))
            if "left_pseudo_disp" in sample.keys():
                sample['left_pseudo_disp'] = np.copy(np.flipud(sample['left_pseudo_disp']))
            if "right_pseudo_disp" in sample.keys():
                sample['right_pseudo_disp'] = np.copy(np.flipud(sample['right_pseudo_disp']))
            if "gt_disp" in sample.keys():
                sample['gt_disp'] = np.copy(np.flipud(sample['gt_disp'])) 

        return sample




class ToPILImage(object):
    def __call__(self, sample):
        if 'clear_left' in sample.keys():
            sample['clear_left'] = Image.fromarray(sample['clear_left'].astype('uint8'))
        if 'clear_right' in sample.keys():
            sample['clear_right'] = Image.fromarray(sample['clear_right'].astype('uint8'))

        return sample



class ToNumpyArray(object):
    def __call__(self, sample):        
        if 'clear_left' in sample.keys():
            sample['clear_left'] = np.array(sample['clear_left']).astype(np.float32)
        if 'clear_right' in sample.keys():
            sample['clear_right'] = np.array(sample['clear_right']).astype(np.float32)
        
        return sample




# Random coloring
class RandomContrast(object):
    """Random contrast"""
    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            if 'clear_left' in sample.keys():
                sample['clear_left'] = F.adjust_contrast(sample['clear_left'], contrast_factor)
            if 'clear_right' in sample.keys():
                sample['clear_right'] = F.adjust_contrast(sample['clear_right'], contrast_factor)
            
        return sample




class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet
            if 'clear_left' in sample.keys():
                sample['clear_left'] = F.adjust_gamma(sample['clear_left'], gamma)
            if 'clear_right' in sample.keys():
                sample['clear_right'] = F.adjust_gamma(sample['clear_right'], gamma)
                
        return sample



class RandomBrightness(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)
            if 'clear_left' in sample.keys():
                sample['clear_left'] = F.adjust_brightness(sample['clear_left'], brightness)
            if 'clear_right' in sample.keys():
                sample['clear_right'] = F.adjust_brightness(sample['clear_right'], brightness)

        return sample




class RandomHue(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)
            if 'clear_left' in sample.keys():
                sample['clear_left'] = F.adjust_hue(sample['clear_left'], hue)
            if 'clear_right' in sample.keys():
                sample['clear_right'] = F.adjust_hue(sample['clear_right'], hue)

        return sample



class RandomSaturation(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            if 'clear_left' in sample.keys():
                sample['clear_left'] = F.adjust_saturation(sample['clear_left'], saturation)
            if 'clear_right' in sample.keys():
                sample['clear_right'] = F.adjust_saturation(sample['clear_right'], saturation)
        
        return sample



class RandomColor(object):
    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample