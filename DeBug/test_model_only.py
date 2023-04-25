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
from losses.DSSMDLoss import Disparity_Loss,TransmissionMap_Loss,Airlight_Loss,RecoveredCleanImagesLoss
from models.DSSMD import DSSMMD
from utils.metric import P1_metric,P1_Value,D1_metric,Disparity_EPE_Loss
from utils.visual import save_images,disp_error_img
from dataloader.preprocess import scale_disp


# Keys 
complete_data=['clear_left_image','clear_right_image','left_disp','right_disp',
                                                  'focal_length','baseline','beta','airlight']

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).type_as(x))


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

import os
from utils.AverageMeter import AverageMeter
from utils.common import logger, check_path, write_pfm,count_parameters


def compute_the_psnr(dehaze_image,original_image):
    img_loss = img2mse(dehaze_image,original_image)
    psnr = mse2psnr(img_loss)
    return psnr

def L1_Loss(pred,ground_truth):
    
    b,c,h,w = pred.shape
    diff = torch.abs(pred-ground_truth)
    
    l1_error = diff.sum()/(h*w*b*c)
    
    return l1_error

def L1_Loss_s(pred,ground_truth):
    
    b,c = pred.shape
    diff = torch.abs(pred-ground_truth)
    
    l1_error = diff.sum()/(b*c)
    
    return l1_error
    
# Evaluation Here
def saved_json(outputs,foldername):
    import json
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    sub_folder = os.path.join(foldername,"metrics")
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    if isinstance(outputs,dict):
        with open(os.path.join(sub_folder,"outcome.json"),'w') as f:
            json.dump(outputs,f,indent=4) 

import skimage.io
if __name__=="__main__":
    

    
    root_files = "DSSMD_Results2"
    gt_disp_path = os.path.join(root_files,'gt_disp')
    haze_left_images_path = os.path.join(root_files,"haze_left")
    haze_right_images_path = os.path.join(root_files,"haze_right")
    clear_left_images_path = os.path.join(root_files,'clear_left')
    clear_right_images_path = os.path.join(root_files,"clear_right")
    error_map_path = os.path.join(root_files,'error_map')
    dehaze_left_image_path = os.path.join(root_files,'dehaze_left')
    predicted_disparity_path = os.path.join(root_files,'predicted_disparity')
    predicted_trans_path = os.path.join(root_files,"predicted_trans")
    
    saved_metris = os.path.join(root_files,'metric')

    # make new files
    if not os.path.exists(root_files):
        os.makedirs(root_files)
    if not os.path.exists(haze_left_images_path):
        os.makedirs(haze_left_images_path)
    if not os.path.exists(haze_right_images_path):
        os.makedirs(haze_right_images_path)
    if not os.path.exists(clear_left_images_path):
        os.makedirs(clear_left_images_path)
    if not os.path.exists(clear_right_images_path):
        os.makedirs(clear_right_images_path)
    if not os.path.exists(error_map_path):
        os.makedirs(error_map_path)
    if not os.path.exists(dehaze_left_image_path):
        os.makedirs(dehaze_left_image_path)
    if not os.path.exists(predicted_disparity_path):
        os.makedirs(predicted_disparity_path)
    if not os.path.exists(gt_disp_path):
        os.makedirs(gt_disp_path)
    if not os.path.exists(predicted_trans_path):
        os.makedirs(predicted_trans_path)
    
    if not os.path.exists(saved_metris):
        os.makedirs(saved_metris)
        
    

    # Loss Function Designe
    disp_EPEs = AverageMeter()
    P1_meter = AverageMeter()

    

    
    
    
    
    pretrained_net = DSSMMD(in_channels=3,dehaze_switch=False)
    
    pretrained_net = torch.nn.DataParallel(pretrained_net, device_ids=[0]).cuda()
    
    pretrained_path = "/home/zliu/Desktop/StereoDehazing_Reimp/DSSMD/DSSMD_Disp_Only_1.989.pth"
    ckpt = torch.load(pretrained_path)
    
    pretrained_net.load_state_dict(ckpt["state_dict"])
    pretrained_net.eval()
    
    print("Load Successfully")   
    
    
    
    
    
    # FILE PATH.
    file_path = "/media/zliu/datagrid1/liu/sceneflow"
    
    # TRAIN LIST AND VAL LIST.
    train_list = "../filenames/SceneFlow_Fog.list"
    val_list = "../filenames/SceneFlow_Fog_Val.list"
    
    
    train_loader,test_loader = prepare_dataset(file_path=file_path,train_list=train_list,val_list=val_list)

    
    for i, sample_batched in enumerate(test_loader):
        
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
        right_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=right_disp)

        left_depth_l = F.interpolate(left_depth,size=[clear_left_image.shape[-2],clear_left_image.shape[-1]],mode='bilinear',
                                        align_corners=False)
        right_depth_l = F.interpolate(right_depth,size=[clear_left_image.shape[-2],clear_left_image.shape[-1]],mode='bilinear',
                                        align_corners=False)
        haze_left = recover_haze_images(clean_images=clear_left_image,beta=beta,A=airlight,depth=left_depth_l)
        haze_right = recover_haze_images(clean_images=clear_right_image,beta=beta,A=airlight,depth=right_depth_l)
        
        # haze_left = recover_haze_images(clean_images=clear_left_image,beta=beta,A=airlight,depth=left_depth)
        # haze_right = recover_haze_images(clean_images=clear_right_image,beta=beta,A=airlight,depth=right_depth)
        
        trans_left = depth2trans(left_depth_l,beta=beta)
        trans_right = depth2trans(right_depth_l,beta=beta)
        
        # print(trans_left.shape)
        
        
        with torch.no_grad():
            disparity_pyramid = pretrained_net(haze_left,haze_right)
            output = disparity_pyramid[-1]
            # print(output.shape)

            # dehaze_image = RecoveredCleanFromTrans(pred_trans,pred_airlight,haze_left)
            output = scale_disp(output, (output.size()[0], 540, 960))
            error_maps = disp_error_img(output.squeeze(1),left_disp.squeeze(1))
            
            
            disp_epe = Disparity_EPE_Loss(output,left_disp)
            p1_errpr = P1_metric(output,left_disp)
            # psnr_cur = compute_the_psnr(dehaze_image,clear_left_image)
            # trans_cur = L1_Loss(pred_trans,trans_left)
            # airlight_cur = L1_Loss_s(pred_airlight,airlight)
            
            
            disp_EPEs.update(disp_epe.data.item(),clear_left_image.size(0))
            P1_meter.update(p1_errpr.data.item(),clear_left_image.size(0))
            # psnr_meters.update(psnr_cur.data.item(),clear_left_image.size(0))
            # transmision_L1_meter.update(trans_cur.data.item(),clear_left_image.size(0))
            # airlight_L1_meter.update(airlight_cur.data.item(),clear_left_image.size(0))
            
        
        # # to visible numpys
        if i%50==0:
            print("Epoch {}/{}".format(i,len(test_loader)))
            predicted_disparity_vis = output.squeeze(0).squeeze(0).cpu().numpy()
            gt_disparity_left_vis = left_disp.squeeze(0).squeeze(0).cpu().numpy()
            left_image_vis = haze_left.squeeze(0).permute(1,2,0).cpu().numpy()
            right_image_vis = haze_right.squeeze(0).permute(1,2,0).cpu().numpy()
            clear_left_vis = clear_left_image.squeeze(0).permute(1,2,0).cpu().numpy()
            clear_right_vis = clear_right_image.squeeze(0).permute(1,2,0).cpu().numpy()
            # deahze_left_vis = dehaze_image.squeeze(0).permute(1,2,0).cpu().numpy()
            error_maps_vis = error_maps.squeeze(0).permute(1,2,0).cpu().numpy()
            # predicted_trans_vis = pred_trans.squeeze(0).permute(1,2,0).cpu().numpy()
            
            skimage.io.imsave(os.path.join(predicted_disparity_path,'{}.png'.format(i)),predicted_disparity_vis)
            skimage.io.imsave(os.path.join(gt_disp_path,'{}.png'.format(i)),gt_disparity_left_vis)
            skimage.io.imsave(os.path.join(haze_left_images_path,'{}.png'.format(i)),left_image_vis)
            skimage.io.imsave(os.path.join(haze_right_images_path,'{}.png'.format(i)),right_image_vis)
            skimage.io.imsave(os.path.join(clear_left_images_path,'{}.png'.format(i)),clear_left_vis)
            skimage.io.imsave(os.path.join(clear_right_images_path,'{}.png'.format(i)),clear_right_vis)
            # skimage.io.imsave(os.path.join(dehaze_left_image_path,'{}.png'.format(i)),deahze_left_vis)
            skimage.io.imsave(os.path.join(error_map_path,'{}.png'.format(i)),error_maps_vis)
            # skimage.io.imsave(os.path.join(predicted_trans_path,'{}.png'.format(i)),predicted_trans_vis)
        
    

    
    output_result = dict()
    output_result['EPE'] = disp_EPEs.avg
    output_result["P1_error"] = P1_meter.avg
    
    saved_json(outputs=output_result,foldername=saved_metris)
    