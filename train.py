import os 

import numpy 
import torch 

from data import download_usps, get_usps, _default_path
from model import Model

if not os.path.exists(_default_path): 
        download_usps()
ds = get_usps() 

def _rescale_img(img: numpy.array): 
    # quoting the paper: The gray levels of each image are scaled and 
    # translated to fall within the range -1 to 1. 
    return img / 127.5 - 1.0 

_sample_idx = 0 
_usps_size = len(ds['train']) 
def next_sample():
    x = numpy.array(ds['train'][_sample_idx]['image'], dtype=numpy.float32)
    x = _rescale_img(x)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)

    y = torch.ones(10) * -1 
    label = ds['train'][_sample_idx]['label']
    y[label] = 1

    sample_idx += 1 
    if sample_idx > _usps_size - 1:  
        sample_idx = 0 
        
    return x, y  

if __name__ == '__main__': 
    model = Model()
    x, y = next_sample()
    res = model.__call__(x, y)
    print(y)
    print(res)