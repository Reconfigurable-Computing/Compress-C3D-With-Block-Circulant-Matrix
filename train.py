
import timeit
import os
from tqdm import tqdm
import torch
#torch.cuda.current_device()
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloaders.dataset import VideoDataset
from network import C3D_model,CirC3D,SpectralCirC3D
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import argparse
import os
from utils import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

parser=argparse.ArgumentParser(description="training C3D model")
parser.add_argument('--nEpochs',type=int,default=200)
parser.add_argument('--resume_epoch',type=int,default=0)
parser.add_argument('--test_interval',type=int,default=10)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--dataset',type=str,default='ucf101',help='ucf101 or hmdb51')
parser.add_argument('--num_workers',type=int,default=4)
parser.add_argument('--step_size',type=int,default=20)
parser.add_argument('--gamma',type=float,default=0.5)
parser.add_argument('--weight_decay',type=float,default=5e-4)
parser.add_argument('--model_type',type=str,default="norm",help='norm or circulant or qspectralcir,default=norm')
parser.add_argument('--block_size',type=int,default=8,help="循环矩阵分块大小")
cfg=parser.parse_args()

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = cfg.nEpochs
resume_epoch = cfg.resume_epoch
test_interval = cfg.test_interval
lr = cfg.lr
batch_size=cfg.batch_size
dataset = cfg.dataset
best_acc=0


if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(save_dir_root, 'run')

def train_model():
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    global best_acc
    #model select
    if cfg.model_type=="norm":
        print("using C3D")
        model=C3D_model.C3D(101)
    elif cfg.model_type=="circulant":
        print("using CirC3D with block_size={}".format(cfg.block_size))
        model=CirC3D.CirC3D(101,block_size=cfg.block_size)
    elif cfg.model_type=='qspectralcir':
        print("using QSpectralCirC3D with block_size={}".format(cfg.block_size))
        model=SpectralCirC3D.SpectralCirC3D(101,block_size=cfg.block_size)
    else:
        raise NotImplementedError
    print(model)
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    #
    model=model.to(device)
    #model=model.cuda()
    criterion=criterion.to(device)
    #criterion=criterion.cuda()
    model = torch.nn.DataParallel(model)
    if cfg.resume_epoch == 0:
        print("Training from scratch...")
    '''else:
        ckpt=torch.load(os.path.join(save_dir,"checkpoint_R{}_B{}.pth".format(cfg.depth,cfg.block_size)))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        cfg.resume_epoch=ckpt['epoch']
        best_acc=ckpt['acc']
        print("Load last epoch data:last epoch={},acc={}%".format(cfg.resume_epoch,best_acc))'''
    #
    print('Training model on {} dataset...'.format(cfg.dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=cfg.dataset, split='train',clip_len=16), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dataloader   = DataLoader(VideoDataset(dataset=cfg.dataset, split='val',  clip_len=16), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_dataloader  = DataLoader(VideoDataset(dataset=cfg.dataset, split='test', clip_len=16), batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}

    for epoch in range(cfg.resume_epoch, cfg.nEpochs):
        print("\nEposh: {}".format(epoch))
        #
        print("training...")
        model.train()
        model.mode=True
        adjust_learning_rate(optimizer=optimizer,current_epoch=epoch,max_epoch=cfg.nEpochs,lr_min=0,lr_max=cfg.lr,warmup=True)
        print("current epoch's learning rate is {}".format(optimizer.param_groups[0]['lr']))
        total_train_loss=0
        total_train_correct=0
        total_train_num=0
        train_loss=0.0
        correct=0
        total=0
        train_interval=10
        for batch_idx,(inputs,targets) in enumerate(trainval_loaders['train']):
            inputs,targets=inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,targets.long())
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            total_train_loss+=loss.item()
            _,preds=outputs.max(1)
            total+=targets.size(0)
            total_train_num+=targets.size(0)
            correct+=targets.eq(preds).sum().item()
            total_train_correct+=targets.eq(preds).sum().item()
            #
            if batch_idx % train_interval == train_interval-1:
                print("Epoch={},loss={}".format(epoch,train_loss/train_interval))
                print("Epoch={},acc={}%".format(epoch,correct/total*100.0))
                train_loss=0.0
                correct=0
                total=0
        print("Epoch {}'s Total Training loss:{}".format(epoch,total_train_loss / (total_train_num / cfg.batch_size)))
        print("Epoch {}'s Total Training acc:{}%".format(epoch,total_train_correct / total_train_num * 100.0))
        #scheduler.step()
        #val
        print("validating...")
        model.eval()
        model.mode=False
        val_loss=0.0
        correct=0
        total=0
        with torch.no_grad():
            for batch_idx,(inputs,targets) in enumerate(trainval_loaders['val']):
                inputs,targets=inputs.to(device),targets.to(device)
                outputs=model(inputs)
                loss=criterion(outputs,targets.long())
                _,preds=outputs.max(1)
                val_loss+=loss.item()
                total+=targets.size(0)
                correct+=preds.eq(targets).sum().item()
        acc=100.0*correct/total
        print("Epoch={},val loss={}".format(epoch,val_loss/(total/cfg.batch_size)))
        print("Epoch={},val acc={}%".format(epoch,acc))
        #
        if acc>best_acc:
            #����ģ��
            print("saving...")
            state={
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'acc':acc,
                'epoch':epoch,
            }
            torch.save(state,os.path.join(save_dir,"checkpoint_{}_B{}.pth".format(cfg.model_type,cfg.block_size)))
            best_acc=acc
        #test
        '''
        if epoch%cfg.test_interval==cfg.test_interval-1:
            model.eval()
            test_loss=0.0
            correct=0
            total=0
            print("testing...")
            with torch.no_grad():
                for batch_idx,(inputs,targets) in enumerate(test_dataloader):
                    inputs,targets=inputs.to(device),targets.to(device)
                    outputs=model(inputs)
                    loss=criterion(outputs,targets.long())
                    _,preds=torch.max(outputs,dim=1)
                    test_loss+=loss.item()
                    total+=targets.size(0)
                    correct+=preds.eq(targets).sum().item()
        #
            print("test loss={}".format(test_loss/(total/cfg.batch_size)))
            print("test acc={}%".format(correct/total*100.0))
        ''' 


if __name__ == "__main__":
    train_model()