import torch.nn as nn
from .common import Conv2
import torch
from .attention import make_attention  # edit

def make_cnet(cfg, in_chans):
    return ContextNet([in_chans]+cfg.dims)

class ContextNet(nn.Module):
    def __init__(self,c) -> None:
        super().__init__()
        # self.d = d
        self.down0 = Conv2(c[0],c[1])
        # self.reduce0 = conv(c[0],d[0])
        # self.attnet0 = make_attention(c[1]) # edit
        self.down1 = Conv2(c[1],c[2])
        # self.reduce1 = conv(c[1],d[1])
        # self.attnet1 = make_attention(c[2]) # edit
        self.down2 = Conv2(c[2],c[3])
        # self.reduce2 = conv(c[2],d[2])
        self.attnet2 = make_attention(c[3]) # edit

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        
        x = self.down0(x)
        # x = self.attnet0(x) # edit
        x1 = x
        x = self.down1(x)
        # x = self.attnet1(x) # edit
        x2 = x
        x = self.down2(x)
        x = self.attnet2(x) # edit
        x3 = x

        if is_list:
            x1 = torch.split(x1, [batch_dim, batch_dim], dim=0)
            x2 = torch.split(x2, [batch_dim, batch_dim], dim=0)
            x3 = torch.split(x3, [batch_dim, batch_dim], dim=0)
            return [x1[0],x2[0],x3[0]], [x1[1],x2[1],x3[1]]

        return [x1,x2,x3]