import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockCirConv3d(nn.Module):
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
        return F.conv3d(input=x, weight=ori_weight, bias=self.bias, stride=self.stride, padding=self.padding)



if __name__=='__main__':
    pass
    # M=24
    # N=16
    # H=32
    # W=32
    # D=16
    # K=3
    # S=1
    # P=1
    # B=8
    # block_size=8
    # #
    # circonv3d=BlockCirConv3d(in_channels=N,out_channels=M,stride=S,padding=P,kernel_size=K,block_size=block_size)
    # x=torch.randn((B,N,D,H,W))
    # w=torch.randn((M//block_size,N//block_size,block_size,K,K,K))
    # b=torch.randn((M,))
    # circonv3d.weight.data=w
    # circonv3d.bias.data=b
    # y=circonv3d.forward(x)
    # print(y.size())
    # #
    # z=CirConv3d(x=x,w=w,b=b,stride=S,padding=P,block_size=block_size)
    # print(z.size())
    # print(torch.max(torch.abs(z-y)))