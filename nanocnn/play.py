import torch
import numpy as np 
from PIL import Image  

from model import Model
from train import _rescale_img

def _load_nums() -> list[np.array]: 
    tile = 16 
    img = Image.open('../numbers_16x16.png').convert('L')
    w,h = img.size 

    nums = []
    for i in range(0, int(w/tile)): 
        num = img.crop(box=(i*tile, 0, i*tile + tile, tile))
        num = np.array(num, dtype=np.float32)
        num = _rescale_img(num)
        num = torch.from_numpy(num)
        num = num.unsqueeze(0)
        nums.append(num)
        

    return nums

if __name__ == '__main__': 
    nums = _load_nums() 
    model = Model().from_pretrained()
    model.eval()
    for i, num in enumerate(nums): 
        res = model.__call__(num, None) 
        print(f'gt: {i} pred: {res[0].argmax()} out: {torch.softmax(res[0], dim=-1)}')
