import torch
import torch.nn as nn
import torch.nn.functional as F

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.complex(F.relu(x.real),F.relu(x.imag))

class SBN(nn.Module):
    def __init__(self,block_size,num_features):
        super().__init__()
        self.block_size=block_size
        self.num_features=(num_features//self.block_size)*(self.block_size//2+1)                  #考虑共轭对称性后的实际通道数
        #self.num_features=num_features
        self.bn_real=nn.BatchNorm3d(num_features=self.num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn_imag=nn.BatchNorm3d(num_features=self.num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self,x):
        batch,channel,frame,height,width=x.size()                               
        assert channel==self.num_features
        y=torch.complex(self.bn_real(x.real),self.bn_imag(x.imag))
        return y.contiguous().view(batch,channel,frame,height,width)
        
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

class SpectralPool(nn.Module):
    def __init__(self,block_size,kernel_size,stride,padding):
        super().__init__()
        self.block_size=block_size
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding

    def forward(self,x):
        out_real=F.max_pool3d(input=x.real, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        out_imag=F.max_pool3d(input=x.imag, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return torch.complex(out_real,out_imag)