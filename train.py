import os 

import numpy 
import torch 

from data import download_usps, get_usps, _default_path

if not os.path.exists(_default_path): 
        download_usps()
ds = get_usps() 

def _rescale_img(img: numpy.array): 
    """
    quoting the paper: The gray levels of each image are scaled and 
    translated to fall within the range -1 to 1. 
    """
    return img / 127.5 - 1.0 

def _pad_img(img: numpy.array, pad: int = 3): 
     """
     quoting the paper: Connections extending past the boundaries of the
     input plane take their input from a virtual background plane whose state
     is equal to a constant, pretedetermined background level, in our case -1. 
     """
     return numpy.pad(img, ((0, pad), (0, pad)), mode='constant', constant_values=-1)

_sample_idx = 0 
def next_sample():
    x = numpy.array(ds['train'][_sample_idx]['image'])
    x = _rescale_img(x)
    x = _pad_img(x)
    y = torch.zeros(10)
    label = ds['train'][_sample_idx]['label']
    y[label] = 1
    return x, y  

if __name__ == '__main__': 
    print(ds.keys())
    print(ds['train'])
    print(ds['test'])
    x, y = next_sample()
    print(x) 
    print(y)