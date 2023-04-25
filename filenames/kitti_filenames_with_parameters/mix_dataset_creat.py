import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("../..")
from utils.utils import read_text_lines
import random


def Get_Rest(list1,list2):
    rest = []
    for item in list1:
        if item not in list2:
            rest.append(item)
    return rest

if __name__=="__main__":
    
    kitti2015_complete_path = "kitti2015_complete_normal_pseudo_para.txt"
    kitti2012_complete_path = "kitti2012_complete_normal_pseudo_para.txt"
    
    
    kitti2012_lines = read_text_lines(kitti2012_complete_path)
    kitti2015_lines = read_text_lines(kitti2015_complete_path)
    
    kitti12_nums = len(kitti2012_lines)
    kitti15_nums = len(kitti2015_lines)
    
    kitti12_sample_train = random.sample(kitti2012_lines,int(0.95*kitti12_nums))
    kitti12_sample_val = Get_Rest(kitti2012_lines,kitti12_sample_train)

    kitti15_sample_train = random.sample(kitti2015_lines,int(0.95*kitti15_nums))
    kitti15_sample_val = Get_Rest(kitti2015_lines,kitti15_sample_train)

    
    kitti15_sample_train.extend(kitti12_sample_train)
    kitti15_sample_val.extend(kitti12_sample_val)
    
    
    
    with open("kitti_mix_train_normal_pseudo_para.txt",'w') as f:
        for idx, line in enumerate(kitti15_sample_train):
            if "colored" in line:
                addv_before = 'kitti_2012'
            else:
                addv_before = 'kitti_2015'
            splits = line.split()
            left = splits[0]
            left = os.path.join(addv_before,left)
            right = splits[1]
            right = os.path.join(addv_before,right)
            disp = splits[2]
            disp = os.path.join(addv_before,disp)
            left_pseudo = splits[3]
            left_pseudo = os.path.join(addv_before,left_pseudo)
            right_pseudo = splits[4]
            right_pseudo = os.path.join(addv_before,right_pseudo)
            
            baseline = splits[5]
            focal_length = splits[6]
            beta = splits[7]
            airlight = splits[8]
            
            line = left + " " + right + " " + disp + " " + left_pseudo + " " + right_pseudo + " " +str(baseline) +" "+ str(focal_length) +" "+    \
            str(beta) + " " + str(airlight)
             
            if idx!=len(kitti15_sample_train)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
                
    
    with open("kitti_mix_val_normal_pseudo_para.txt",'w') as f:
        for idx, line in enumerate(kitti15_sample_val):
            
            if "colored" in line:
                addv_before = 'kitti_2012'
            else:
                addv_before = 'kitti_2015'
            splits = line.split()
            left = splits[0]
            left = os.path.join(addv_before,left)
            right = splits[1]
            right = os.path.join(addv_before,right)
            disp = splits[2]
            disp = os.path.join(addv_before,disp)
            left_pseudo = splits[3]
            left_pseudo = os.path.join(addv_before,left_pseudo)
            right_pseudo = splits[4]
            right_pseudo = os.path.join(addv_before,right_pseudo)
            
            baseline = splits[5]
            focal_length = splits[6]
            beta = splits[7]
            airlight = splits[8]
            
            line = left + " " + right + " " + disp + " " + left_pseudo + " " + right_pseudo + " " +str(baseline) +" "+ str(focal_length) +" "+    \
            str(beta) + " " + str(airlight)
            if idx!=len(kitti15_sample_val)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)