import torch
import torch.nn as nn
import torch.nn.functional as F 

import os
import sys

sys.path.append("..")
from dataloader.ComplexSceneflowLoader import ComplexStereoDataset
from dataloader import complex_transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.devtools import convert_tensor_to_image
from losses.DSSMDLoss import RecoveredCleanImagesLoss

# Keys 
complete_data=['clear_left_image','clear_right_image','left_disp','right_disp',
                                                  'focal_length','baseline','beta','airlight']

# SceneFlow dataset
def prepare_dataset(file_path,train_list,val_list):
    test_batch =1
    num_works = 1
    train_transform_list = [complex_transforms.RandomCrop(320, 640),
                            complex_transforms.ToTensor(),
                            # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
    train_transform = complex_transforms.Compose(train_transform_list)    
    
    val_transform_list = [complex_transforms.ToTensor(),
                        #   transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
    val_transform = complex_transforms.Compose(val_transform_list)
    
    train_dataset = ComplexStereoDataset(data_dir=file_path,train_datalist=train_list,test_datalist=val_list,
                                    dataset_name='SceneFlow',mode='train',transform=train_transform,
                                    visible_list=complete_data)

    test_dataset = ComplexStereoDataset(data_dir=file_path,train_datalist=train_list,test_datalist=val_list,
                                    dataset_name='SceneFlow',mode='val',transform=val_transform,
                                    visible_list=complete_data)

    train_loader = DataLoader(train_dataset, batch_size = test_batch, \
                                shuffle = True, num_workers = num_works, \
                                pin_memory = True)
    
    test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                                shuffle = False, num_workers = num_works, \
                                pin_memory = True)
    

    return train_loader,test_loader


def convert_disp_to_depth(baseline,focal_length,disp):
    b,c,h,w = disp.shape
    assert baseline.shape[0] == b
    assert focal_length.shape[0] == b
    baseline = baseline.view(b,1,1,1)
    focal_length = focal_length.view(b,1,1,1)
    depth = focal_length * baseline /(disp)
    return depth

def convert_depth_to_disp(baseline,focal_length,depth):
    b,c,h,w = depth.shape

    assert baseline.shape[0] == b
    assert focal_length.shape[0] == b
    baseline = baseline.view(b,1,1,1)
    focal_length = focal_length.view(b,1,1,1)
    disparity = focal_length * baseline/(depth)
    return disparity

def recover_clear_images(foggy_images,depth,beta,A):
    b,c,h,w = foggy_images.shape
    assert beta.shape[0]==b
    assert A.shape[0]==b
    beta = beta.view(b,1,1,1)
    A = A.view(b,1,1,1)
    '''
    Inputs:     Baseline, focal length, Beta, A, depth
                        Foggy Images.[B,3,H,W], make sure A is in [0-255]
    
    Returns:    Clean Images[B,3,H,W]
    '''
    norm_depth = depth.repeat(1,3,1,1) #[B,3,H,W]
    transmission = torch.exp(-norm_depth*beta)
    A = A
    clear_images = (foggy_images- A*(1-transmission))/(transmission)

    return clear_images

def recover_haze_images(clean_images,beta,A,depth):
    b,c,h,w = clean_images.shape
    assert beta.shape[0]==b
    assert A.shape[0]==b
    beta = beta.view(b,1,1,1)
    A = A.view(b,1,1,1)
    '''
    Inputs:     Baseline, focal length, Beta, A, depth
                        Clean Images.[B,3,H,W], make sure A is in [0-255]
    
    Returns:    Foggy Images[B,3,H,W]
    '''
    norm_depth = depth.repeat(1,3,1,1) #[B,3,H,W]
    transmission = torch.exp(-norm_depth*beta)
    A = A
    foggy_images = (clean_images*transmission) + A*(1-transmission)
    
    return foggy_images    

def recover_depth(clean_images,foggy_images,beta,A):
    b,c,h,w = clean_images.shape
    assert beta.shape[0]==b
    assert A.shape[0]==b
    beta = beta.view(b,1,1,1) #[b,1,1,1]
    A = A.view(b,1,1,1)
    A = A               #[b,1,1,1]
    
    depth = torch.log(torch.abs((foggy_images-A)/(clean_images-A))) * -1.0 /(beta)
    return depth


def depth2trans(depth,beta):
    b,c,h,w = depth.shape
    assert beta.shape[0]==b
    beta = beta.view(b,1,1,1)
    
    norm_depth = depth #[B,3,H,W]
    transmission = torch.exp(-norm_depth*beta)
    
    return transmission

def trans2depth(trans,beta):
    
    b,c,h,w = trans.shape
    assert beta.shape[0]==b
    beta = beta.view(b,1,1,1)
    trans = torch.clamp(trans,min=1e-10)
    depth = torch.log(trans) * -1/beta

    return depth
    
    
    

# Evaluation Here
if __name__=="__main__":
    
    # FILE PATH.
    file_path = "/media/zliu/datagrid1/liu/sceneflow"
    
    # TRAIN LIST AND VAL LIST.
    train_list = "../filenames/SceneFlow_Fog.list"
    val_list = "../filenames/SceneFlow_Fog_Val.list"
    
    
    train_loader,test_loader = prepare_dataset(file_path=file_path,train_list=train_list,val_list=val_list)

    
    for i, sample_batched in enumerate(train_loader):
        
        # clean left
        clear_left_image = Variable(sample_batched['clear_left_image'].cuda(), requires_grad=False)
        # clean right
        clear_right_image = Variable(sample_batched['clear_right_image'].cuda(), requires_grad=False)
        
        # left disp
        left_disp = Variable(sample_batched['left_disp'].cuda(), requires_grad=False).unsqueeze(1)
        # right disp
        right_disp = Variable(sample_batched['right_disp'].cuda(), requires_grad=False).unsqueeze(1)
        # focal length
        focal_length = Variable(sample_batched['focal_length'].cuda(), requires_grad=False)
        # baseline
        baseline = Variable(sample_batched['baseline'].cuda(), requires_grad=False)
        # beta
        beta = Variable(sample_batched['beta'].cuda(), requires_grad=False)
        # airlight
        airlight = Variable(sample_batched['airlight'].cuda(), requires_grad=False)
        
        left_disp[left_disp<0.1] = 0.1
        right_disp[right_disp<0.1] = 0.1
        left_disp[left_disp>192] = 192
        right_disp[right_disp>192] = 192
        
        
        left_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=left_disp)
        right_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=left_disp)
        
        haze_left = recover_haze_images(clean_images=clear_left_image,beta=beta,A=airlight,depth=left_depth)
        haze_right = recover_haze_images(clean_images=clear_right_image,beta=beta,A=airlight,depth=right_depth)
        
        trans_left = depth2trans(left_depth,beta=beta)
        trans_right = depth2trans(right_depth,beta=beta)
        
        
        recover_loss = RecoveredCleanImagesLoss('normal')
        
        loss = recover_loss(trans_left,airlight,haze_left,clear_left_image)
        
        print(loss.mean())
        # print(trans_left.shape)
        # print(clear_left_image.shape)
        # plt.imshow(clear_left_image.squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.axis('off')
        # plt.show()
        
        break


    pass