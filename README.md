# 代码结构
**dataloaders文件夹**：视频数据集解析及读取    
**network**：C3D模型，包括未压缩的C3D模型、分块循环矩阵+全频域+INT8量化感知训练的C3D模型    
**mypath.py**：数据集路径，修改output_dir为视频数据集的路径  
**inference.py**：推理测试(未测试)  
**utils.py**：学习率调整函数  
**train.py**：训练代码
# 使用说明
训练时，通过  
--nEpochs指定训练时的迭代轮数  
--lr指定学习率  
--batch_size指定批处理大小  
--dataset指定数据集，目前仅支持ucf101和hmdb51两个数据集
--num_workers指定线程数  
--weight_decay指定权重衰减系数  
--model_type指定模型类型，norm表示未压缩的C3D模型,qspectralcir表示循环矩阵+全频域+INT8量化感知训练优化后的C3D模型  
--block_size表示循环矩阵分块大小，以此调节压缩比(可以为1,2,4,8,16等2的幂)  

这里  
频域算子采用的是**CReLU**、**CBN**和**CMaxpooling**:分别在实部和虚部作用ReLU、BN、Maxpooling即可。  
INT8量化感知训练采用的是有符号对称量化，量化后的范围为[-127,127],实部虚部分别量化，激活和权重分别量化，采用各自独立的量化伸缩因子，具体可以参考谷歌的INT8量化感知训练。
# 实验记录
## 训练参数:
nEpochs=200,
lr=0.01,
batch_size=64,
dataset=ucf101,
weight_decay=5e-4,
model_type=norm(即原始的未压缩的C3D模型),
block_size=1。   
## 精度:
95.507%       

## 训练参数
nEpochs=300(考虑到压缩后的网络收敛更慢，因此训练300个epochs),
lr=0.01,
batch_size=64,
dataset=ucf101,
weight_decay=1e-4,
model_type=qspectralcir(循环矩阵+全频域+INT8量化感知训练优化后的C3D模型),
block_size=8  
## 精度:
94.257%  
## 结论
接近**32x**的压缩比(8x来自分块循环矩阵压缩，4x来自INT8量化),且压缩后的网络结构十分规整，而精度损失仅为**1.25%**。