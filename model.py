import torch.nn as nn 

class Model(nn.Module): 
    def __init__(
        self, 
        h1_size: int=12,
    ): 
        self.h1 = nn.ModuleList([
            nn.Conv2d(
                # greyscale img 
                in_channels=1, 
                out_channels=1,
                # paper params
                kernel_size=5,
                stride=2
            ) for _ in range(h1_size)
        ])

    def __call__(self): 
        pass 
