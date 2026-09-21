import torch
import torch.nn as nn 

class Model(nn.Module): 
    def __init__(
        self, 
        h1_size: int=12,
        h2_size: int=12
    ): 
        super().__init__()
        self.h1 = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=1,
                kernel_size=5,
                stride=2, 
                bias=True 
            ) for _ in range(h1_size)
        ])

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None): 
        res = self.h1[0](x)
        return res 