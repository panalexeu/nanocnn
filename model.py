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
        self.h2 = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=1, 
                kernel_size=5, 
                stride=2
            ) for _ in range(h2_size)
        ])

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None): 
        # quoting the paper: Connections extending past the boundaries of the
        # input plane take their input from a virtual background plane whose state
        # is equal to a constant, pretedetermined background level, in our case -1.
        x = nn.functional.pad(x, (0, 3, 0, 3), mode='constant', value=-1)
        out = torch.cat([l(x) for l in self.h1], dim=0)
        out = nn.functional.pad(out, (0, 3, 0, 3), mode='constant', value=-1)
        out = torch.cat([l(o.unsqueeze(0)) for l in self.h2 for o in out], dim=0)
        print(out.shape)