#from importlib.metadata import requires
#from asyncio import FastChildWatcher
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def quant(x, scale_r,scale_i):
    q_r=Round.apply(x.real/scale_r)
    q_i=Round.apply(x.imag/scale_i)
    return torch.complex(torch.clamp(q_r,-127,127),torch.clamp(q_i,-127,127))   #对称量化

def dequant(x, scale_r,scale_i):
    return torch.complex(x.real*scale_r,x.imag*scale_i)

#baseline
class SpatialCirConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride,padding,kernel_size,block_size,bias=True):
        super().__init__()
        self.block_size=block_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        assert in_channels%block_size==0
        assert out_channels%block_size==0
        self.o_nblocks=out_channels//block_size
        self.i_nblocks=in_channels//block_size
        #
        weight=torch.empty(self.o_nblocks,self.i_nblocks,self.block_size,kernel_size,kernel_size,kernel_size,requires_grad=True)       #默认是(3,3,3)大小的kernel
        inited_weight = torch.nn.init.xavier_uniform_(weight, gain=1)
        self.weight=torch.nn.Parameter(inited_weight)
        if bias:
            uninit_bias = torch.empty(self.out_channels, requires_grad=True)
            inited_bias = torch.nn.init.constant_(uninit_bias, 0.0)
            self.bias=torch.nn.Parameter(inited_bias)
        else:
            self.bias=None

    def forward(self,x):
        # 根据压缩后权重恢复原始的循环权重
        ori_weight=torch.empty((self.out_channels,self.in_channels,self.kernel_size,self.kernel_size,self.kernel_size),device=x.device)
        for q in range(self.i_nblocks):
            for j in range(self.block_size):
                ori_weight[:, q * self.block_size + j, :, :, :]=torch.cat([self.weight[:,q,-j:,:,:,:],self.weight[:,q,:-j,:,:,:]],dim=1).view(-1,self.kernel_size,self.kernel_size,self.kernel_size)
        # 卷积
        batch, channel, frames, height, width = x.size()
        assert channel == self.in_channels//self.block_size*(self.block_size//2+1)
        o_frame = (frames + 2 * self.padding - self.kernel_size) // self.stride + 1
        o_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        o_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        #IFFT变换回时域
        #print(x.size())
        x_iffted=torch.fft.irfft(x.view(batch,self.in_channels//self.block_size,self.block_size//2+1,frames,height,width),dim=2)\
            .contiguous().view(batch,self.in_channels,frames,height,width)
        #print(x_iffted.size())
        #时域分块循环矩阵卷积
        out = F.conv3d(input=x_iffted, weight=ori_weight, bias=self.bias, stride=self.stride, padding=self.padding)
        #print(out.size())
        #FFT变换回频域
        return torch.fft.rfft(out.view(batch,self.out_channels//self.block_size,self.block_size,o_frame,o_height,o_width),dim=2)\
            .contiguous().view(batch,self.out_channels//self.block_size*(self.block_size//2+1),o_frame,o_height,o_width)

class QSpatialCirConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride,padding,kernel_size,block_size,bias=True):
        super().__init__()
        self.block_size=block_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        assert in_channels%block_size==0
        assert out_channels%block_size==0
        self.o_nblocks=out_channels//block_size
        self.i_nblocks=in_channels//block_size
        self.momentum = 0.1
        self.first_batch = True
        #
        self.register_buffer('scale_w_r', None)
        self.register_buffer('scale_w_i', None)
        self.register_buffer('scale_a_r', None)
        self.register_buffer('scale_a_i', None)
        self.mode=True
        #
        weight=torch.empty(self.o_nblocks,self.i_nblocks,self.block_size,kernel_size,kernel_size,kernel_size,requires_grad=True)       #默认是(3,3,3)大小的kernel
        inited_weight = torch.nn.init.xavier_uniform_(weight, gain=1)
        self.weight=torch.nn.Parameter(inited_weight)
        if bias:
            uninit_bias = torch.empty(self.out_channels, requires_grad=True)
            inited_bias = torch.nn.init.constant_(uninit_bias, 0.0)
            self.bias=torch.nn.Parameter(inited_bias)
        else:
            self.bias=None

    def forward(self,x):
        #对权重进行量化+伪量化操作
        w_ffted=torch.fft.rfft(self.weight,dim=2)
        if self.mode:
            scale_w_real=torch.max(torch.abs(w_ffted.real))/127
            scale_w_imag=torch.max(torch.abs(w_ffted.imag))/127
            self.scale_w_r=scale_w_real
            self.scale_w_i=scale_w_imag
        wq_ffted=dequant(quant(w_ffted,self.scale_w_r,self.scale_w_i),self.scale_w_r,self.scale_w_i)
        self.weight.data=torch.fft.irfft(wq_ffted,dim=2).contiguous()
        #
        # 根据压缩后权重恢复原始的循环权重
        ori_weight=torch.empty((self.out_channels,self.in_channels,self.kernel_size,self.kernel_size,self.kernel_size),device=x.device)
        for q in range(self.i_nblocks):
            for j in range(self.block_size):
                ori_weight[:, q * self.block_size + j, :, :, :]=torch.cat([self.weight[:,q,-j:,:,:,:],self.weight[:,q,:-j,:,:,:]],dim=1).view(-1,self.kernel_size,self.kernel_size,self.kernel_size)
        # 卷积
        batch, channel, frames, height, width = x.size()
        assert channel == self.in_channels//self.block_size*(self.block_size//2+1)
        o_frame = (frames + 2 * self.padding - self.kernel_size) // self.stride + 1
        o_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        o_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        #IFFT变换回时域
        ##对输入进行量化+伪量化操作
        if self.mode:
            scale_a_real=torch.max(torch.abs(x.real))/127
            scale_a_imag=torch.max(torch.abs(x.imag))/127                            #实部虚部分别计算得到一个scale
            if self.first_batch:
                self.scale_a_r=scale_a_real
                self.scale_a_i=scale_a_imag
                self.first_batch=False
            else:
                self.scale_a_r=scale_a_real*self.momentum+self.scale_a_r*(1-self.momentum)
                self.scale_a_i=scale_a_imag*self.momentum+self.scale_a_i*(1-self.momentum)
        #
        x=dequant(quant(x,self.scale_a_r,self.scale_a_i),self.scale_a_r,self.scale_a_i)
        ##
        x_iffted=torch.fft.irfft(x.view(batch,self.in_channels//self.block_size,self.block_size//2+1,frames,height,width),dim=2)\
            .contiguous().view(batch,self.in_channels,frames,height,width)
        #print(x_iffted.size())
        #时域分块循环矩阵卷积
        out = F.conv3d(input=x_iffted, weight=ori_weight, bias=self.bias, stride=self.stride, padding=self.padding)
        #FFT变换回频域
        return torch.fft.rfft(out.view(batch,self.out_channels//self.block_size,self.block_size,o_frame,o_height,o_width),dim=2)\
            .contiguous().view(batch,self.out_channels//self.block_size*(self.block_size//2+1),o_frame,o_height,o_width)


class FFT(nn.Module):
    def __init__(self,block_size):
        super().__init__()
        self.block_size=block_size

    def forward(self,x):
        batch,channel,frame,height,width=x.size()
        x_ffted=torch.fft.rfft(x.view(batch,channel//self.block_size,self.block_size,frame,height,width),dim=2)
        return x_ffted.contiguous().view(batch,channel//self.block_size*(self.block_size//2+1),frame,height,width)

class IFFT(nn.Module):
    def __init__(self,block_size):
        super().__init__()
        self.block_size=block_size

    def forward(self,x):
        batch,channel,frame,height,width=x.size()                               
        x_iffted=torch.fft.irfft(x.view(batch,channel//(self.block_size//2+1),self.block_size//2+1,frame,height,width),dim=2)
        return x_iffted.contiguous().view(batch,channel//(self.block_size//2+1)*self.block_size,frame,height,width)

        
if __name__=='__main__':
    
    M=32
    N=24
    B=8
    D=16
    H=24
    W=24
    K=3
    S=1
    P=1
    #
    x=torch.randn(1,N,D,H,W)
    w=torch.randn(M//B,N//B,B,K,K,K)
    print(w.size())
    conv=QSpatialCirConv(in_channels=N,out_channels=M,stride=S,padding=P,kernel_size=K,block_size=B,bias=False)
    conv.weight.data=w
    x_ffted=torch.fft.rfft(x.view(1,N//B,B,D,H,W),dim=2).contiguous().view(1,-1,D,H,W)
    print(x_ffted.size())
    y=conv(x_ffted)
    print(y.size())