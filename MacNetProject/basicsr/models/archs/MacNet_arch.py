import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import sys
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
import numbers


from timm.models.layers import DropPath
from functools import partial
from typing import Callable

from basicsr.models.archs.SS2D_arch import SS2D


from einops import rearrange
import os
sys.path.append(os.getcwd())



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x.permute(0, 3, 1, 2))


class DConv7(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)

    def forward(self, x):
        return self.dconv(x)


class PConv(nn.Module):
    def __init__(self, f_number,excitation_factor=2) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

class HDAM(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(HDAM, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        self.dwconv2 = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=n_fea_in, bias=True)
        self.dwconv3 = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=n_fea_in, bias=True)
        self.dconv7 = DConv7(n_fea_middle)
        self.pconv = PConv(n_fea_middle)
    
    def forward(self, img):
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea1 = self.depth_conv(x_1)
        illu_fea2 = self.dwconv2(x_1)
        illu_fea3 = self.dwconv3(x_1)
        illu_fea = illu_fea1 + illu_fea2 + illu_fea3
        illu_fea = self.pconv(illu_fea)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map
    

class R_D(nn.Module):
    def __init__(self, num=64):
        super(R_D, self).__init__()
        

        self.R_D = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            )   
        self.finalconv = nn.Sequential(           
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        x = self.R_D(input) 
        x = self.finalconv(x) + input
        return torch.sigmoid(x)
    
class L_D(nn.Module):
    def __init__(self, num=64):
        super(L_D, self).__init__()
        self.L_D = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_D(input))
    
    

class GF1(nn.Module):
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(GF1, self).__init__()

        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm2(dim)
        self.attn = GF(dim, num_heads, bias)
        self.norm2 = LayerNorm2(dim)
        self.ffn = IEL(dim)

    def forward(self, input_R, input_S):
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))
        return input_R



class IG(nn.Module):
    def __init__(self, dim = 16, dim_head=16, heads=2, num_blocks=1,d_state = 16):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                GF1(dim_2=16,dim = 16,num_heads = heads,ffn_expansion_factor=2.66 ,bias=True, LayerNorm_type='WithBias'),
                SS2D(d_model=dim,dropout=0,d_state =d_state),   
                PreNorm(dim, IEL(dim=dim))  
            ]))

    def forward(self, x, illu_fea):
      
        for (trans,ss2d,ff) in self.blocks:
            y=trans(x,illu_fea).permute(0, 2, 3, 1)
            x=ss2d(y)+ x.permute(0, 2, 3, 1)  
            x = ff(x) + x.permute(0, 3, 1, 2)
        return x
    

class SIAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.L_net = L_D(num=64)
        self.initial_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.IGuide = IG()
        

    def forward(self, x,illu_fea):

        x = self.initial_conv(x)
        x = F.relu(x)
        illu_map2 = self.initial_conv(illu_fea)
        x4 = self.IGuide(x, illu_map2)
        x4 = F.relu(x4)
        x1 = self.L_net(x4)
        return x4,x1

class LayerNorm2(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False,norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.ln_1 = norm_layer(hidden_features)
        self.self_attention = SS2D(d_model=hidden_features,dropout=0,d_state =16)
        self.drop_path = DropPath(0)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        input_right = x1.permute(0,2,3,1)
        input_right = self.ln_1(input_right)
        input_right = self.self_attention(input_right)
        input_right = self.drop_path(input_right)
        input_right = input_right.permute(0,3,1,2)
        x1 = self.Tanh(input_right) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
    
class GF(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GF, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ADI_L(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(ADI_L, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm2(dim)
        self.ffn = GF(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
    
class ADI_R(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(ADI_R, self).__init__()
        self.norm = LayerNorm2(dim)
        self.gdfn = IEL(dim)
        self.ffn = GF(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x
    


class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm2(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x

class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm2(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
            
    def forward(self, x,y):
        x = self.up_scale(x)
        x = torch.cat([x, y],dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x
        


class CACM(nn.Module):
    def __init__(self, 
                 channels=[40, 40, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CACM, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.LE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.LE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.LE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.LE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.LD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.LD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.LD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.LD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=0,bias=False)
        )
        
        
        self.RE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.RE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.RE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.RE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.RD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.RD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.RD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.RD_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=0,bias=False),
            )
        
        self.ADI_L1 = ADI_L(ch2, head2)
        self.ADI_L2 = ADI_L(ch3, head3)
        self.ADI_L3 = ADI_L(ch4, head4)
        self.ADI_L4 = ADI_L(ch4, head4)
        self.ADI_L5 = ADI_L(ch3, head3)
        self.ADI_L6 = ADI_L(ch2, head2)

        
        self.ADI_R1 = ADI_R(ch2, head2)
        self.ADI_R2 = ADI_R(ch3, head3)
        self.ADI_R3 = ADI_R(ch4, head4)
        self.ADI_R4 = ADI_R(ch4, head4)
        self.ADI_R5 = ADI_R(ch3, head3)
        self.ADI_R6 = ADI_R(ch2, head2)
        self.dconv7 = DConv7(40)
        self.pconv = PConv(40)
        self.conv1 = nn.Conv2d(3, 40, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            40, 40, kernel_size=5, padding=2, bias=True, groups=4)

        self.conv2 = nn.Conv2d(40, 3, kernel_size=1, bias=True)
        self.dwconv2 = nn.Conv2d(40, 40, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=4, bias=True)
        self.dwconv3 = nn.Conv2d(40, 40, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=4, bias=True)
        
        
        
        
    def forward(self, x,y,z):
        
        # low
        i_enc0 = self.RE_block0(x)
        i_enc1 = self.RE_block1(i_enc0)
        hv_0 = self.LE_block0(y)
        hv_1 = self.LE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.ADI_R1(i_enc1, hv_1)
        hv_2 = self.ADI_L1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_mix2 = hv_2 + i_enc2
        i_enc2 = self.RE_block2(i_mix2)
        hv_L = self.LE_block2(hv_1)
        i_L = self.RE_block1(z)
        i_L = self.RE_block2(i_L)


        
        i_enc3 = self.ADI_R2(i_enc2, hv_L)
        hv_3 = self.ADI_L2(hv_L, i_enc2)
        v_jump2 = i_enc3
        
        hv_jump2 = hv_3
        i_mix3 = hv_3 + i_enc3 
        i_enc3 = self.RE_block3(i_mix3)
        i_L = self.RE_block3(i_L)
        
        i_enc4 = self.ADI_R3(i_enc3, i_L)
        hv_4 = self.ADI_L3(i_L, i_enc3)
        
        i_dec4 = self.ADI_R4(i_enc4,hv_4)
        hv_4 = self.ADI_L4(hv_4, i_enc4)
        
        hv_3 = self.LD_block3(hv_4, hv_jump2)
        i_dec3 = self.RD_block3(i_dec4, v_jump2)
        i_dec2 = self.ADI_R5(i_dec3, hv_3)
        hv_2 = self.ADI_L5(hv_3, i_dec3)
        
        hv_2 = self.LD_block2(hv_2, hv_jump1)
        i_dec2 = self.RD_block2(i_dec3, v_jump1)
        
        i_dec1 = self.ADI_R6(i_dec2, hv_2)
        hv_1 = self.ADI_L6(hv_2, i_dec2)
        
        i_dec1 = self.RD_block1(i_dec1, i_jump0)
        i_dec0 = self.RD_block0(i_dec1)
        hv_1 = self.LD_block1(hv_1, hv_jump0)
        hv_0 = self.LD_block0(hv_1)
        out = hv_0 + i_dec0
        
        
        

        return out
    

class MacNet_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(MacNet_Single_Stage, self).__init__()
        self.R_D = R_D(num=64)
        self.SIAM = SIAM()
        self.HDAM = HDAM(n_feat)
        self.CACM = CACM()
        
    def forward(self, img,gt):

        alpha = 4 
        beta = 50    
        img_bright = img * alpha + beta / 255.0
        R1 = self.R_D(gt)
        illu_fea, illu_map = self.HDAM(img)
        input_img = img * illu_map + img
        R2 = self.R_D(img_bright)
        L1,L2 = self.SIAM(gt,R1)
        L3,L4= self.SIAM(input_img,img_bright)
        I = L4 * R2
        output_img = self.CACM(input_img,I,illu_fea)
        return output_img,L1,L2,L3,L4,R1,R2
    
class MacNet_Single_Stage1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(MacNet_Single_Stage1, self).__init__()
        self.R_D = R_D(num=64)
        self.SIAM = SIAM()
        self.HDAM = HDAM(n_feat)
        self.denoiser = CACM()

    
    def forward(self, img):
        alpha = 4  
        beta = 50    
        img_bright = img * alpha + beta / 255.0
        illu_fea, illu_map = self.HDAM(img)
        input_img = img * illu_map + img
        R2 = self.R_D(img_bright)
        L3,L4= self.SIAM(input_img,img_bright)
        I = L4 * R2
        output_img = self.denoiser(input_img,I,illu_fea)
        return output_img,L3,L4,R2


# Multi-stage MacNet without GT (inference/training)
class MacNet1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super(MacNet1, self).__init__()
        self.stage = stage
        self.body = nn.ModuleList([
            MacNet_Single_Stage1(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
            for _ in range(stage)
        ])

    def forward(self, xx):
        outputs = [] 

        for stage in self.body:

            out, L3, L4, R2= stage(xx)



            xx = out

        return out, L3, L4, R2
    

# Multi-stage MacNet with GT (training)  
class MacNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super(MacNet, self).__init__()
        self.stage = stage

        self.body = nn.ModuleList([
            MacNet_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
            for _ in range(stage)
        ])

    def forward(self, xx, gt):
        outputs = []  

        for stage in self.body:
            out, L1, L2, L3, L4, R1, R2= stage(xx, gt)

            xx = out

        # 返回所有阶段的输出结果
        return out, L1, L2, L3, L4, R1, R2






from fvcore.nn import FlopCountAnalysis



if __name__ == '__main__':
    model = MacNet1(stage=1, n_feat=40, num_blocks=[1, 2, 2]).cuda()
    
    print(model)
    input_img = torch.randn((1, 3, 128 ,128)).cuda()  
    gt_img = torch.randn((1, 3, 128 ,128)).cuda()    

    output_img, L3, L4, R2 = model(input_img)

    flops = FlopCountAnalysis(model, (input_img))
    
    n_param = sum([p.nelement() for p in model.parameters()]) 
    
    print(f'GMac: {flops.total()/(1024*1024*1024)}')
    print(f'Params: {n_param}')
