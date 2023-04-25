import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from models.BidNet.STTM_Simple import STTM_Single
from models.UtilsNet.submoudles import *


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class AirlightBasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.layer = nn.Sequential(
           nn.Conv2d(input_dim,output_dim,kernel_size=3,stride=1,padding=1,bias=False),
           nn.BatchNorm2d(output_dim),
           nn.LeakyReLU(0.2),
           nn.AvgPool2d(kernel_size=2,stride=2)
        )
    
    def forward(self,x):        
        return self.layer(x)


    
class BidNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stereo Macthing Branch Here
        self.conv1 = conv(3,16,3,1) # 1
        self.conv2 = conv(16,32,3,1)
        
        self.conv3 = ResBlock(32, 64, stride=2)            # 1/2
        self.conv4 = ResBlock(64, 128, stride=2)           # 1/4
        
        self.conv5 = ResBlock(128, 256, stride=2)           # 1/8
        self.conv5_1 = ResBlock(256, 256)
        self.conv6 = ResBlock(256, 256, stride=2)           # 1/16
        self.conv6_1 = ResBlock(256, 256)
        self.conv7 = ResBlock(256, 512, stride=2)          # 1/32
        self.conv7_1 = ResBlock(512, 512)
        
        self.iconv5 = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(96,32, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(64,32, 3, 1, 1)

        
        self.upconv5 = deconv(512, 256)
        self.upconv4 = deconv(256, 128)
        self.upconv3 = deconv(128, 64)
        self.upconv2 = deconv(64, 32)
        self.upconv1 = deconv(32,32) # Note there is 32 dimension
        
        
        self.stm = STTM_Single(dim=32,heads=1,dim_head=32,out_dim=32)

        self.transmission_estimation_branch = nn.Sequential(
            nn.Conv2d(32,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,1,3,1,1,bias=False),
            nn.Sigmoid()
        )
        
        # Airlight estimation Branch
        self.preprocess = conv(in_planes=3,out_planes=16,kernel_size=3,stride=1)
        self.encoder1 = AirlightBasicBlock(16,16)
        self.encoder2 = AirlightBasicBlock(16,16)
        self.encoder3 = AirlightBasicBlock(16,16)
        
        self.decoder3_up = deconv(16,16)
        self.decoder3_iconv = nn.ConvTranspose2d(32, 16, 3, 1, 1)
        self.decoder2_up = deconv(16,16)
        self.decoder2_iconv = nn.ConvTranspose2d(32, 16, 3, 1, 1)
        self.decoder1_up = deconv(16,16)
        self.decoder1_iconv = nn.ConvTranspose2d(32, 16, 3, 1, 1)
        self.airlight_estimation = nn.Sequential(
            nn.Conv2d(16,1,3,1,1,bias=False),
            nn.Sigmoid()
        )
        
        # SPP layer for refinement
        self.branch1 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     nn.Conv2d(1, 16, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     nn.Conv2d(1, 16, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True))
        
        self.branch3 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2, 2)),
                                     nn.Conv2d(1, 16, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True))
        
        self.refinement = nn.Sequential(nn.Conv2d(16+16+16+1,out_channels=16,kernel_size=3,padding=1,stride=1,bias=False),
                                        nn.BatchNorm2d(16),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(16,1,3,1,1,bias=False),
                                        nn.Sigmoid())

    def forward(self,left_img,right_img):
        
        # Unet Architecture
        # left feature
        left_conv1 = self.conv1(left_img)
        right_conv1 = self.conv1(right_img) # 3-->16
        
        left_conv2 = self.conv2(left_conv1)
        right_conv2 = self.conv2(right_conv1) # 16-->32  
        
        left_conv3 = self.conv3(left_conv2)   # 32-->64 1/2
        right_conv3 = self.conv3(right_conv2)
        
        left_conv4 = self.conv4(left_conv3)
        right_conv4 = self.conv4(right_conv3) # 64-->128 1/4
        
        left_conv5 = self.conv5(left_conv4)   
        left_conv5_1 = self.conv5_1(left_conv5)    # 128-->256 1/8
        
        right_conv5 = self.conv5(right_conv4)
        right_conv5_1 = self.conv5_1(right_conv5)
        
        left_conv6 = self.conv6(left_conv5_1) # 256-->256 1/16
        left_conv6_1 = self.conv6_1(left_conv6) 
        right_conv6 = self.conv6(left_conv5_1)
        right_conv6_1 = self.conv6_1(right_conv6)
        
        left_conv7 = self.conv7(left_conv6_1)  # 256-->512 1/32
        left_conv7_1 = self.conv7_1(left_conv7)
        right_conv7 = self.conv7(right_conv6_1)
        right_conv7_1 = self.conv7_1(right_conv7)        
        
        # upsample 1/16 resolution.
        upconv_left6 = self.upconv5(left_conv7_1) 
        concated6_left = torch.cat((left_conv6_1,upconv_left6),dim=1)
        iconv6_left = self.iconv5(concated6_left)  # 1/16 , 512-->256
        
        upconv_right6 = self.upconv5(right_conv7_1) 
        concated6_right = torch.cat((right_conv6_1,upconv_right6),dim=1)
        iconv6_right = self.iconv5(concated6_right)  # 1/16 , 512-->256
        
        # 1/8 resolution
        upconv_left5 = self.upconv4(iconv6_left) 
        concated5_left = torch.cat((left_conv5_1,upconv_left5),dim=1)
        iconv5_left = self.iconv4(concated5_left)  
        upconv_right5 = self.upconv4(iconv6_right) 
        concated5_right = torch.cat((right_conv5_1,upconv_right5),dim=1)
        iconv5_right = self.iconv4(concated5_right)  # 1/8 , 256-->128

        # 1/4 resolution
        upconv_left4 = self.upconv3(iconv5_left) 
        concated4_left = torch.cat((left_conv4,upconv_left4),dim=1)
        iconv4_left = self.iconv3(concated4_left)  #
        upconv_right4 = self.upconv3(iconv5_right) 
        concated4_right = torch.cat((right_conv4,upconv_right4),dim=1)
        iconv4_right = self.iconv3(concated4_right)  # 1/4 , 128-->64
        
        # 1/2 resolution
        upconv_left3 = self.upconv2(iconv4_left) 
        concated3_left = torch.cat((left_conv3,upconv_left3),dim=1)
        iconv3_left = self.iconv2(concated3_left)  #
        
        upconv_right3 = self.upconv2(iconv4_right) 
        concated3_right = torch.cat((right_conv3,upconv_right3),dim=1)
        iconv3_right = self.iconv2(concated3_right)  # 1/2 , 128-->64
        
        # Full resolution
        upconv_left2 = self.upconv1(iconv3_left) 
        concated2_left = torch.cat((left_conv2,upconv_left2),dim=1)
        iconv2_left = self.iconv1(concated2_left)  #
        
        upconv_right2 = self.upconv1(iconv3_right) 
        concated2_right = torch.cat((right_conv2,upconv_right2),dim=1)
        iconv2_right = self.iconv1(concated2_right)  # 1/2 , 128-->64
        
        # STM Module
        left_feat = self.stm(iconv2_left,iconv2_right)
        right_feat = self.stm(iconv2_right,iconv2_left)

        # initial left and right transmission.
        initial_left_trans = self.transmission_estimation_branch(left_feat)
        initial_right_trans = self.transmission_estimation_branch(right_feat)

        # Airlight Estimation
        airlight_feat = self.preprocess(left_img)
        airlight_feat1 = self.encoder1(airlight_feat) # 1/2
        airlight_feat2 = self.encoder2(airlight_feat1) # 1/4
        airlight_feat3 = self.encoder3(airlight_feat2) # 1/8
        
        airlight_up2 = self.decoder3_up(airlight_feat3)
        air_concat2 = torch.cat((airlight_up2,airlight_feat2),dim=1)
        airlight_recover2 = self.decoder3_iconv(air_concat2)

        airlight_up1 = self.decoder2_up(airlight_recover2)
        air_concat1 = torch.cat((airlight_up1,airlight_feat1),dim=1)
        airlight_recover1 = self.decoder2_iconv(air_concat1)

        airlight_up0 = self.decoder1_up(airlight_recover1)
        air_concat0 = torch.cat((airlight_up0,airlight_feat),dim=1)
        airlight_recover0 = self.decoder1_iconv(air_concat0)
        
        # estimated airlight
        estimated_airlight = self.airlight_estimation(airlight_recover0)

        
        # refined transmission map layer: SPP Layer
        spp_1_left = self.branch1(initial_left_trans) # 1/8        
        spp_1_right = self.branch1(initial_right_trans)
        spp_1_left = F.interpolate(spp_1_left,size=[left_img.size(-2),left_img.size(-1)],mode='bilinear',align_corners=False)
        spp_1_right = F.interpolate(spp_1_right,size=[left_img.size(-2),left_img.size(-1)],mode='bilinear',align_corners=False)
        
        spp_2_left = self.branch2(initial_left_trans) # 1/4       
        spp_2_right = self.branch2(initial_right_trans)
        spp_2_left = F.interpolate(spp_2_left,size=[left_img.size(-2),left_img.size(-1)],mode='bilinear',align_corners=False)
        spp_2_right = F.interpolate(spp_2_right,size=[left_img.size(-2),left_img.size(-1)],mode='bilinear',align_corners=False)
        
        spp_3_left = self.branch3(initial_left_trans) # 1/2        
        spp_3_right = self.branch3(initial_right_trans)
        spp_3_left = F.interpolate(spp_3_left,size=[left_img.size(-2),left_img.size(-1)],mode='bilinear',align_corners=False)
        spp_3_right = F.interpolate(spp_3_right,size=[left_img.size(-2),left_img.size(-1)],mode='bilinear',align_corners=False)
        
        spp_concat_left = torch.cat([initial_left_trans,spp_1_left,spp_2_left,spp_3_left],dim=1)
        spp_concat_right = torch.cat([initial_right_trans,spp_1_right,spp_2_right,spp_3_right],dim=1)
        
        refined_trans_left = self.refinement(spp_concat_left)
        refined_trans_right = self.refinement(spp_concat_right)
        
        return initial_left_trans,initial_right_trans,refined_trans_left,refined_trans_right,estimated_airlight





if __name__=="__main__":
    
    left= torch.randn(1,3,320,640).cuda()
    right = torch.randn(1,3,320,640).cuda()
    
    bident = BidNet().cuda()
    
    left_trans_init,right_trans_init,refined_trans_left,refined_trans_right,estimated_airlight=bident(left,right)
    
    
    pass
