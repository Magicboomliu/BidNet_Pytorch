import torch
import numpy as np


def convert_tensor_to_image(tensor):
    dim = tensor.shape[1]
    if dim==3:
        image = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    elif dim==1:
        image = tensor.squeeze(0).squeeze(0).cpu().numpy()
        
    return image
        