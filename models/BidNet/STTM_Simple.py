import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


class STTM_Single(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,out_dim=256):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5


        self.attend = nn.Softmax(dim = -1)

        
        self.q = nn.Conv2d(dim,inner_dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.k = nn.Conv2d(dim,inner_dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.v = nn.Conv2d(dim,inner_dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_out =nn.Sequential(
                nn.Conv2d(inner_dim*2,out_dim,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_dim,out_dim,kernel_size=1,stride=1,padding=0,bias=False)
            )
            
        
        
    def forward(self,left_feat,right_feat):

        q = self.q(left_feat)
        k = self.k(right_feat)
        v = self.v(right_feat)
        b,c,h,w = q.shape
        
        
        q = q.permute(0,2,3,1)
        k = k.permute(0,2,3,1)
        v = v.permute(0,2,3,1)

        q = q.reshape(b*h,w,c).contiguous()
        k = k.reshape(b*h,w,c).contiguous()
        v = v.reshape(b*h,w,c).contiguous()
        
        dots = torch.matmul(q,k.transpose(-1, -2))*self.scale
        
        attn = self.attend(dots)
        
        out = torch.matmul(attn,v)
        out = out.view(b,h,w,c).contiguous()
        out = out.permute(0,3,1,2)
        
        
        new_left = torch.cat((left_feat,out),dim=1)
        
        output = self.to_out(new_left)
        
        return output      
 


if __name__=="__main__":
    
    left = torch.randn(8,128,40,80).cuda()
    right = torch.randn(8,128,40,80).cuda()
    
    sttm = STTM_Single(dim=128,heads=4,dim_head=32,out_dim=128).cuda()
    
    output = sttm(left,right)
    
    print(output.shape)