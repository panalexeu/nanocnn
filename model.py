from typing import Self

import torch
import torch.nn as nn 

class Model(nn.Module): 
    def __init__(
        self, 
        h1_size: int = 12,
        h2_size: int = 12, 
        h3_size: int = 30,
    ): 
        super().__init__()

        self.init_factor = 2.4

        self.h1 = nn.Conv2d(
            in_channels=1, 
            out_channels=h1_size,
            kernel_size=5,
            stride=2, 
            bias=False 
        ) 
        h1_in = 5 ** 2 # 25
        torch.nn.init.uniform_(self.h1.weight, a=-self.init_factor / h1_in ** 0.5, b=self.init_factor / h1_in ** 0.5)
        # quoting the paper: Thus layer H1  comprises ... but only 
        # 1068 free parameters (768 biases plus 25 times 12 feature kernels)
        self.h1_bias = nn.Parameter(torch.zeros(h1_size, 8, 8))

        # for now this part is ignored and information is combined for all 12 feature 
        # maps from level H1: Each unit in H2 combines local information coming from 8
        #  of the 12 different feature maps in H1. 
        self.h2 = nn.Conv2d(
            in_channels=h2_size, 
            out_channels=h2_size, 
            kernel_size=5, 
            stride=2, 
            bias=False
        ) 
        h2_in = h1_size * h1_in # 12 * 5 * 5 = 300
        torch.nn.init.uniform_(self.h2.weight, a=-self.init_factor / h2_in ** 0.5, b=self.init_factor / h2_in ** 0.5)
        # quoting the paper: All these connections are controlled by only
        # 2592 free parameters (12 feature maps times 200 weights plus 192 biases). 
        self.h2_bias = nn.Parameter(torch.zeros(h2_size, 4, 4))

        h3_in = h2_size * 16 # 192
        self.h3 = nn.Linear(
            in_features=h3_in,  
            out_features=h3_size,
            bias=True
        )
        torch.nn.init.uniform_(self.h3.weight, a=-self.init_factor / h3_in ** 0.5, b=self.init_factor / h3_in ** 0.5)

        self.out = nn.Linear(
            in_features=h3_size, # 30
            out_features=10,
            bias=True
        ) 
        torch.nn.init.uniform_(self.out.weight, a=-self.init_factor / h3_size ** 0.5, b=self.init_factor / h3_size ** 0.5)

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None = None): 
        # quoting the paper: Connections extending past the boundaries of the
        # input plane take their input from a virtual background plane whose state
        # is equal to a constant, pretedetermined background level, in our case -1.
        x = nn.functional.pad(x, (2, 2, 2, 2), mode='constant', value=-1)
        x = self.h1(x) + self.h1_bias
        x = nn.functional.tanh(x)
        x = nn.functional.pad(x, (2, 2, 2, 2), mode='constant', value=-1)
        x = self.h2(x) + self.h2_bias 
        x = nn.functional.tanh(x)
        x = x.reshape(-1)
        x = self.h3(x)
        x = nn.functional.tanh(x)
        x = self.out(x) 
        x = nn.functional.tanh(x)

        loss = None 
        if target is not None: 
            loss = nn.functional.mse_loss(x, target)

        return x, loss 

    def configure_optimizer(self, lr=0.03): 
        return torch.optim.SGD(self.parameters(), lr=lr)

    @classmethod
    def from_pretrained(cls, ckpt_path: str = './ckpt.pt') -> Self: 
        ckpt = torch.load(ckpt_path)
        model = cls()
        model.load_state_dict(ckpt)
        return model 
    