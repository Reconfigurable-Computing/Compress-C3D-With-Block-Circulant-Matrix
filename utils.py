import matplotlib.pyplot as plt
import math
import torch
from torchvision.models import resnet18
from math import cos, pi
 
def adjust_learning_rate(optimizer, current_epoch,max_epoch,lr_min=0,lr_max=0.1,warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch ==0:
        lr=0.0001                                           #防止第一个epoch学习率为0
    elif current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__=='__main__':
    model = resnet18(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
     
    lr_max=0.1
    lr_min=0.00001
    max_epoch=200
    lrs=[]
    for epoch in range(max_epoch):
        adjust_learning_rate(optimizer=optimizer,current_epoch=epoch,max_epoch=max_epoch,lr_min=lr_min,lr_max=lr_max,warmup=True)
        print(optimizer.param_groups[0]['lr'])
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
     
    plt.plot(lrs)
    plt.show()