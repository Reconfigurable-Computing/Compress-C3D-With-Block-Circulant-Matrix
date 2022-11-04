import torch
import torch.nn as nn
#from mypath import Path
from .CirConv import *
#from .CirFC import *
class CirC3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes,block_size):
        super().__init__()
        self.block_size=block_size

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(num_features=64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        #self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2=BlockCirConv3d(in_channels=64,out_channels=128,stride=1,padding=1,kernel_size=3,block_size=self.block_size)
        self.bn2 = nn.BatchNorm3d(num_features=128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        #self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3a=BlockCirConv3d(in_channels=128,out_channels=256,stride=1,padding=1,kernel_size=3,block_size=self.block_size)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        #self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b=BlockCirConv3d(in_channels=256,out_channels=256,stride=1,padding=1,kernel_size=3,block_size=self.block_size)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        #self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4a=BlockCirConv3d(in_channels=256,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size)
        self.bn4a = nn.BatchNorm3d(num_features=512)
        #self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b=BlockCirConv3d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size)
        self.bn4b = nn.BatchNorm3d(num_features=512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        #self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5a=BlockCirConv3d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size)
        self.bn5a = nn.BatchNorm3d(num_features=512)
        #self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b=BlockCirConv3d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3,block_size=self.block_size)
        self.bn5b = nn.BatchNorm3d(num_features=512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        #self.fc6=CirFC(in_features=8192,out_features=4096,block_size=4096)
        #self.fc7=CirFC(in_features=4096,out_features=4096,block_size=4096)
        #self.fc6 = nn.Linear(8192, 4096)
        #self.fc7 = nn.Linear(4096, 4096)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)

        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.relu(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)

        x = self.avgpool(x)
        x = x.view(-1, 512)
        logits = self.fc(x)

        return logits



if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = CirC3D(num_classes=101,block_size=8)

    outputs = net.forward(inputs)
    print(outputs.size())