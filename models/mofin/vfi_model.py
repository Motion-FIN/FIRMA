import torch.nn as nn
import torch
from .extractor import BasicEncoder
from .flow_up import make_flowup

from validate.bucket import Bucket
from .flow_gen import make_flowgen
from .synth import make_synth
from .context import make_cnet
from .flow_tea import make_flowtea
from .corr import make_corr_fn
import torch.nn.functional as F
# from .attention_encoder import make_attention_encoder
# from einops import rearrange, reduce, repeat
# from .attention import make_attention
# from .attention_encoder import make_attention_encoder
# from einops import rearrange, reduce, repeat

class VFIModel(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.flowgen = make_flowgen(cfg.flowgen)
        
        self.fnet = BasicEncoder(input_dim=3,output_dim=128)
        # self.attnet = make_attention(cfg.att.in_dim)
        # # self.att_ennet = make_attention_encoder(cfg.atten) 
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.corr_fn = make_corr_fn(cfg.corr)

        self.cnet = make_cnet(cfg.cnet,3)
        self.flowup = make_flowup(cfg.flowup)

        self.flowtea = make_flowtea(cfg.flowtea)
        self.synth = make_synth(cfg.synth)

    def forward(self,im0,im1,gt=None,bkt:Bucket=None):
        
        down_scaled = -1 # -1: undecided; 0: no; 1: yes
        down_scale_th = self.cfg.flowgen.down_scale_th

        if self.training or down_scale_th==-1:
            down_scaled = 0

        if down_scaled == -1:
            s = 2
            down0 = F.avg_pool2d(im0,kernel_size=s,stride=s,padding=0,count_include_pad=False)
            down1 = F.avg_pool2d(im1,kernel_size=s,stride=s,padding=0,count_include_pad=False)
            _,_,Hd,Wd = down0.shape

            mod = self.cfg.size_mod

            if Hd%mod>0 or Wd%mod>0:
                pad = ((mod-Hd%mod)%mod,(mod-Wd%mod)%mod)
                down0 = F.pad(down0,(0,pad[1],0,pad[0]))
                down1 = F.pad(down1,(0,pad[1],0,pad[0]))
            else:
                pad = None

            (f0,_),(f1,_) = self.fnet([down0,down1])
            self.corr_fn.setup(f0,f1)
            c0, c1 = self.cnet([down0,down1])
            h, flow, flowgen_flow_list = self.flowgen(down0,down1,self.corr_fn,c0,c1,bkt=bkt)

            if torch.max(flow)*8*s < down_scale_th:
                down_scaled = 0
            else:
                # h,im0,im1,flow,mask,bkt:Bucket=None
                flow, mask, res, flowup_flow_list = self.flowup(h,down0,down1,flow,None,c0,c1,bkt=bkt)

                if pad!=None:
                    flow = flow[:,:,0:Hd,0:Wd]
                    mask = mask[:,:,0:Hd,0:Wd]

                flow = s*F.interpolate(flow,scale_factor=s,mode='bilinear',align_corners=False)
                mask = F.interpolate(mask,scale_factor=s,mode='bilinear',align_corners=False)

                c0, c1 = self.cnet([im0,im1])
                res, mask = self.synth(im0,im1,flow,mask,res,c0,c1)
        
        if down_scaled == 0:
            # # ///////////////////////////////////////////////////////
            # patch_size = 16 #64 # 230807 학습마치고 조정해 볼 것.

            # B0,_,H0,W0 = im0.shape
            # B1,_,H1,W1 = im1.shape

            # attEncIm0 = rearrange(im0, 'b c (h s1) (w s2) -> (h w) b (s1 s2 c)', s1=patch_size, s2=patch_size)
            # attEncIm1 = rearrange(im1, 'b c (h s1) (w s2) -> (h w) b (s1 s2 c)', s1=patch_size, s2=patch_size)

            # att_im0 = self.att_ennet(attEncIm0)
            # att_im1 = self.att_ennet(attEncIm1)
            
            # # att_im0 = attEncIm0
            # # att_im1 = attEncIm1

            # # 1. Rearrange to 2D format
            # encoded_att_im0 = rearrange(att_im0, '(h w) b (s1 s2 c) -> b c (h s1) (w s2)', h=int(H0/patch_size), w=int(W0/patch_size), s1=patch_size, s2=patch_size)
            # encoded_att_im1 = rearrange(att_im1, '(h w) b (s1 s2 c) -> b c (h s1) (w s2)', h=int(H1/patch_size), w=int(W1/patch_size), s1=patch_size, s2=patch_size)
            
            # im01 = encoded_att_im0
            # im11 = encoded_att_im1

            # # 추가 230724
            # im01 = F.relu(self.conv1(im01))
            # im01 = F.relu(self.conv2(im01))
            # im01 = F.relu(self.conv3(im01))

            # f0_ = im01

            # im11 = F.relu(self.conv1(im11))
            # im11 = F.relu(self.conv2(im11))
            # im11 = F.relu(self.conv3(im11))

            # f1_ = im11

            # (f0,_),(f1,_) = self.fnet([im0,im1]) # original
            
            # f0 = (f0 + f0_)/2
            # f1 = (f1 + f1_)/2
            # # ////////////////////////////////////////////////////////////
            (f0,_),(f1,_) = self.fnet([im0,im1])
            # # self.attention score ///////////////////////////////////////
            # f0_ = self.attnet(f0)
            # f1_ = self.attnet(f1)
            
            # f0 = (f0*0.9 + f0_*0.1)
            # f1 = (f1*0.9 + f1_*0.1)
            # # ////////////////////////////////////////////////////////////
            self.corr_fn.setup(f0,f1)
            h, flow, flowgen_flow_list = self.flowgen(im0,im1,self.corr_fn,None,None,bkt=bkt)
            c0, c1 = self.cnet([im0,im1])
            # c0 = [self.attnet0(c0[0]),self.attnet1(c0[1]),self.attnet2(c0[2])] # edit
            # c1 = [self.attnet0(c1[0]),self.attnet1(c1[1]),self.attnet2(c1[2])] # edit
            # print(f'c[0].shpae:{c0[0].shape} || c[1].shape:{c0[1].shape} || c[2].shape:{c0[2].shape}')
            # h,im0,im1,flow,mask,bkt:Bucket=None
            flow, mask, res, flowup_flow_list = self.flowup(h,im0,im1,flow,None,c0,c1,bkt=bkt)
            res, mask = self.synth(im0,im1,flow,mask,res,c0,c1)


        if bkt:
            bkt.push_namespace('tea')
        if gt!=None and self.flowtea!=None:
            output_tea = self.flowtea(im0,im1,flow,mask,gt,flowgen_flow_list+flowup_flow_list,bkt=bkt)
        else:
            output_tea = {}
        if bkt:
            bkt.pop_namespace()

        mask = torch.softmax(mask,1)[:,[0],:,:]

        if bkt != None:
            bkt.push_namespace('r3')
            bkt.new_gray(mask,'mask')
            bkt.pop_namespace()
                
        return flow,mask,res,output_tea