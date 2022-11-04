import torch
import torch.nn as nn
from .CirConv import *
from .QCirconv import QSpatialCirConv
from .SpectralOp import *

class SpectralCirC3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes,block_size):
        super().__init__()
        self.block_size=block_size

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(num_features=64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        #
        self.conv2=QSpatialCirConv(in_channels=64,out_channels=128,stride=1,padding=1,kernel_size=3,block_size=self.block_size,bias=False)
        self.bn2=SBN(block_size=self.block_size,num_features=128)
        self.pool2=SpectralPool(block_size=self.block_size,kernel_size=(2,2,2),stride=(2,2,2),padding=(0,0,0))
        #
        self.conv3a=QSpatialCirConv(in_channels=128,out_channels=256,stride=1,padding=1,kernel_size=3,block_size=self.block_size,bias=False)
        self.bn3a=SBN(block_size=self.block_size,num_features=256)
        self.conv3b=QSpatialCirConv(in_channels=256,out_channels=256,stride=1,padding=1,kernel_size=3,block_size=self.block_size,bias=False)
        self.bn3b=SBN(block_size=self.block_size,num_features=256)
        self.pool3=SpectralPool(block_size=self.block_size,kernel_size=(2,2,2),stride=(2,2,2),padding=(0,0,0))
        #
        self.conv4a=QSpatialCirConv(in_channels=256,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size,bias=False)
        self.bn4a=SBN(block_size=self.block_size,num_features=512)
        self.conv4b=QSpatialCirConv(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size,bias=False)
        self.bn4b=SBN(block_size=self.block_size,num_features=512)
        self.pool4=SpectralPool(block_size=self.block_size,kernel_size=(2,2,2),stride=(2,2,2),padding=(0,0,0))
        #
        self.conv5a=QSpatialCirConv(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size,bias=False)
        self.bn5a=SBN(block_size=self.block_size,num_features=512)
        self.conv5b=QSpatialCirConv(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size,bias=False)
        self.bn5b=SBN(block_size=self.block_size,num_features=512)
        self.pool5=SpectralPool(block_size=self.block_size,kernel_size=(2,2,2),stride=(2,2,2),padding=(0,1,1))
        #
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.fft=FFT(block_size=self.block_size)
        self.ifft=IFFT(block_size=self.block_size)
        self.crelu=CReLU()
        self.relu=nn.ReLU()

    def forward(self, x):
        #时域
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        #频域
        x = self.fft(x)

        x = self.crelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.crelu(self.bn3a(self.conv3a(x)))
        x = self.crelu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        x = self.crelu(self.bn4a(self.conv4a(x)))
        x = self.crelu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)

        x = self.crelu(self.bn5a(self.conv5a(x)))
        x = self.crelu(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)

        x = self.ifft(x)
        #时域

        x = self.avgpool(x)

        x = x.view(-1, 512)
        logits = self.fc(x)

        return logits



if __name__ == "__main__":
    inputs = torch.rand(10, 3, 16, 112, 112)
    net = SpectralCirC3D(num_classes=101,block_size=8)
    outputs = net.forward(inputs)
    print(outputs.size())