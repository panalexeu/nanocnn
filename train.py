import os 

import numpy 
import torch 

from data import download_usps, get_usps, _default_path

if not os.path.exists(_default_path): 
        download_usps()
ds = get_usps() 

_sample_idx = 0 
def next_sample():
    x = numpy.array(ds['train'][_sample_idx]['image'])
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