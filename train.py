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
_train_size = len(ds['train']) 
def next_sample():
    global _sample_idx

    x = numpy.array(ds['train'][_sample_idx]['image'], dtype=numpy.float32)
    x = _rescale_img(x)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)

    y = torch.ones(10) * -1 
    label = ds['train'][_sample_idx]['label']
    y[label] = 1

    _sample_idx += 1 
    if _sample_idx > _train_size - 1:  
        _sample_idx = 0 

    return x, y  

_ema_loss = None
_ema_alpha = 0.001 
def ema(loss: float) -> float: 
     return _ema_alpha * loss + (1-_ema_alpha) * _ema_loss

if __name__ == '__main__': 
    model = Model()
    optimizer = model.configure_optimizer()
    epochs = 23 
    train_steps = epochs * _train_size 
    loggin_steps = 1_000 

    for i in range(train_steps): 
        optimizer.zero_grad()
        _, loss = model.__call__(*next_sample())
        loss.backward()
        _ema_loss = loss.item() if _ema_loss is None else _ema_loss
        _ema_loss = ema(loss.item())
        optimizer.step()

        if i % loggin_steps == 0: 
             print(f'mse loss, step {i}: {loss.item():.4f} ema mse loss: {_ema_loss:.4f}')
         