import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channel):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_channel
        
        self.query_conv = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in//8 , kernel_size= 1).cuda()
        self.key_conv = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in//8 , kernel_size= 1).cuda()
        self.value_conv = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in , kernel_size= 1).cuda()
        self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x.to(out.device)

        return out