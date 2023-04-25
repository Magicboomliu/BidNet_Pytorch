
 

import os
import sys
sys.path.append("..")

from dataloader.ComplexSceneflowLoader import ComplexStereoDataset
from dataloader import complex_transforms

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.devtools import convert_tensor_to_image
from DeBug.inference import convert_disp_to_depth,convert_depth_to_disp,recover_depth,recover_clear_images,recover_haze_images,depth2trans,trans2depth
# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# def convert_disp_to_depth(baseline,focal_length,disp):
#     b,c,h,w = disp.shape
#     assert baseline.shape[0] == b
#     assert focal_length.shape[0] == b
#     baseline = baseline.view(b,1,1,1)
#     focal_length = focal_length.view(b,1,1,1)
#     depth = focal_length * baseline /(disp)
#     return depth


# def convert_depth_to_disp(baseline,focal_length,depth):
#     b,c,h,w = depth.shape

#     assert baseline.shape[0] == b
#     assert focal_length.shape[0] == b
#     baseline = baseline.view(b,1,1,1)
#     focal_length = focal_length.view(b,1,1,1)
#     disparity = focal_length * baseline/(depth)
#     return disparity


# def recover_clear_images(foggy_images,depth,beta,A):
#     b,c,h,w = foggy_images.shape
#     assert beta.shape[0]==b
#     assert A.shape[0]==b
#     beta = beta.view(b,1,1,1)
#     A = A.view(b,1,1,1)
#     '''
#     Inputs:     Baseline, focal length, Beta, A, depth
#                         Foggy Images.[B,3,H,W], make sure A is in [0-255]
    
#     Returns:    Clean Images[B,3,H,W]
#     '''
#     norm_depth = depth.repeat(1,3,1,1) #[B,3,H,W]
#     transmission = torch.exp(-norm_depth*beta)
#     A = A
#     clear_images = (foggy_images- A*(1-transmission))/(transmission)

#     return clear_images


# def recover_haze_images(clean_images,beta,A,depth):
#     b,c,h,w = clean_images.shape
#     assert beta.shape[0]==b
#     assert A.shape[0]==b
#     beta = beta.view(b,1,1,1)
#     A = A.view(b,1,1,1)
#     '''
#     Inputs:     Baseline, focal length, Beta, A, depth
#                         Clean Images.[B,3,H,W], make sure A is in [0-255]
    
#     Returns:    Foggy Images[B,3,H,W]
#     '''
#     norm_depth = depth.repeat(1,3,1,1) #[B,3,H,W]
#     transmission = torch.exp(-norm_depth*beta)
#     A = A
#     foggy_images = (clean_images*transmission) + A*(1-transmission)
    
#     return foggy_images    

    
# def recover_depth(clean_images,foggy_images,beta,A):
#     b,c,h,w = clean_images.shape
#     assert beta.shape[0]==b
#     assert A.shape[0]==b
#     beta = beta.view(b,1,1,1) #[b,1,1,1]
#     A = A.view(b,1,1,1)
#     A = A               #[b,1,1,1]
    
#     depth = torch.log(torch.abs((foggy_images-A)/(clean_images-A))) * -1.0 /(beta)
    
#     return depth


# Keys 
complete_data=['clear_left_image','clear_right_image','left_disp','right_disp',
                                                  'focal_length','baseline','beta','airlight']

def RecoveredCleanFromTrans(transmission_map,airlight,haze_image):
    
    '''
    transmision: [B,1,H,W]
    haze image: [B,3,H,W]
    airlight: [B,1]
    '''
    airlight = airlight.unsqueeze(-1).unsqueeze(-1) #[B,1,1,1]
    
    # if transmission = 0, how to real with?
    if transmission_map.min==0:
        recovered_clean = (haze_image-airlight*(1-transmission_map))/(transmission_map+1e-4)
    else:
        recovered_clean = (haze_image-airlight*(1-transmission_map))/(transmission_map)
        
    recovered_clean = torch.clamp(recovered_clean,min=0,max=1.0)
    
    return recovered_clean


# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# SceneFlow dataset
def prepare_dataset(file_path,train_list,val_list):
    test_batch =8
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



import torch.nn as nn
from losses.DSSMDLoss import RecoveredCleanImagesLossV2
# Evaluation Here
if __name__=="__main__":
    
    
    
    # FILE PATH.
    file_path = "/media/zliu/datagrid1/liu/sceneflow"
    # TRAIN LIST AND VAL LIST.
    train_list = "/home/zliu/Desktop/WeatherStereo/Code/WeatherStereo/Preprocess/SceneFlow_Fog.list"
    val_list = "/home/zliu/Desktop/WeatherStereo/Code/WeatherStereo/Preprocess/SceneFlow_Fog_Val.list"
    
    
    train_loader,test_loader = prepare_dataset(file_path=file_path,train_list=train_list,val_list=val_list)
    transmission_loss = nn.L1Loss(size_average=True,reduction='mean')
    airlight_loss = nn.L1Loss(size_average=True,reduction='mean')   
    recovered_image_loss =  RecoveredCleanImagesLossV2(type='normal')
    
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
        
        left_disp[left_disp<0.1] =0.1
        right_disp[right_disp<0.1] =0.1
        left_disp[left_disp>192] = 192
        right_disp[right_disp>192] = 192
        
        # print(left_disp.shape)
        left_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=left_disp)
        haze_image_left = recover_haze_images(clean_images=clear_left_image,beta=beta,A=airlight,depth=left_depth)
        
        right_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=right_disp)
        haze_image_right = recover_haze_images(clean_images=clear_right_image,beta=beta,A=airlight,depth=right_depth)

        
        left_trans = depth2trans(left_depth,beta=beta)
        right_trans = depth2trans(right_depth,beta=beta)
        
        left_trans_init = left_trans
        refined_trans_left = left_trans
        
        right_trans_init = right_trans
        refined_trans_right = right_trans

        gt_airlight = airlight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        estimated_airlight = gt_airlight
        b,c,h,w = left_depth.shape
        estimated_airlight = estimated_airlight.repeat(1,1,h,w)
        

        haze_left = haze_image_left
        haze_right = haze_image_right
        
        clear_left = clear_left_image
        clear_right = clear_right_image        
        
        
        # transmission loss here
        initial_trans_loss_left = transmission_loss(left_trans_init,left_trans)
        initial_trans_loss_right = transmission_loss(right_trans_init,right_trans)
        refined_trans_loss_left = transmission_loss(refined_trans_left,left_trans)
        refined_trans_loss_right = transmission_loss(refined_trans_right,right_trans)
        transLoss = initial_trans_loss_left + initial_trans_loss_right + refined_trans_loss_left + refined_trans_loss_right
        
        print(transLoss.mean())
        # airlight loss here
        airloss = airlight_loss(estimated_airlight,gt_airlight)
        
        print(airloss.mean())
        
        
        
        # recover image loss here    
        recover_left_initial = recovered_image_loss(left_trans_init,estimated_airlight,haze_left,clear_left)
        recover_left_refine = recovered_image_loss(refined_trans_left,estimated_airlight,haze_left,clear_left)
        recover_right_initial = recovered_image_loss(right_trans_init,estimated_airlight,haze_right,clear_right)
        recover_right_refine = recovered_image_loss(refined_trans_right,estimated_airlight,haze_right,clear_right)
        recover_loss = recover_left_initial + recover_left_refine + recover_right_initial + recover_right_refine

        

        
        total_loss = transLoss + airloss + recover_loss

        print(total_loss)
        break
        
        # recovered_right_image = recover_clear_images(foggy_images=haze_image_right,depth=right_depth,beta=beta,
        #                                              A=airlight)
        
        # print((recovered_right_image-clear_right_image).mean())