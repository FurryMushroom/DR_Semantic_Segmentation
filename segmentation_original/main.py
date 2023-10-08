import torch
import unet
import dataset
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE=5
net = unet.UNet().cuda()
optimizer = torch.optim.Adam(net.parameters()) # 实现Adam算法。Adam: A Method for Stochastic Optimization。
# params (iterable) – 待优化参数的iterable或者是定义了参数组的dict lr (float, 可选) – 学习率（默认：1e-3）
# betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
# eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8） weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
loss_func = nn.BCELoss()
data = dataset.SEGData()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
# PyTorch中深度学习训练的流程是这样的： 1. 创建Dateset 2. Dataset传递给DataLoader 3. DataLoader迭代产生训练数据提供给模型
# Dataset负责建立索引到样本的映射，DataLoader负责以特定的方式从数据集中迭代的产生 一个个batch的样本集合。
# 在enumerate过程中实际上是dataloader按照其参数sampler规定的策略调用了其dataset的getitem方法。
# dataset (Dataset) – 定义好的Map式或者Iterable式数据集。 batch_size (python:int, optional) – 一个batch含有多少样本 (default: 1)。
# shuffle (bool, optional) – 每一个epoch的batch样本是相同还是随机 (default: False)。
# sampler 重点参数，采样器，是一个迭代器
summary = SummaryWriter(r'Log') # 第一个参数 log_dir : 用以保存summary的位置 加r保持路径在读取时不被错读，取消转义 否则双反斜杠\\或正斜杠
"""
Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
The`SummaryWriter`class provides a high-level API to create an event file in a given directory and add summaries 
and events to it. The class updates the file contents asynchronously. This allows a training program to call methods
to add data to the file directly from the training loop, without slowing down training.
"""
EPOCH = 100
print('load net')
# net.load_state_dict(torch.load('SAVE/Unet.pt'))
# 将预训练的参数权重加载到新的模型之中.当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合；如果新构建的模型在层数上进行了部分微调，则上述代码就会报错：key对应不上。
# 如果strict=False,训练权重中与新构建网络中匹配层的键值就进行使用，没有的就默认初始化。
print('load success')
for epoch in range(EPOCH):
    print('begin {} epoch'.format(epoch))
    net.train() # model.train() 将 self.training = True，model.eval() 将 self.training = False，
    # 所以即便训练与测试共用一个模型，也能通过 self.training 来区分现在属于训练还是测试。它们用于切换模式。
    for i, (img, label) in enumerate(dataloader):
        img = img.cuda()
        label = label.cuda()
        img_out = net(img) # 前向传播 forward是在__call__中调用的，而__call__函数是在类的对象使用‘()’时被调用。
        # 此处相当于c++中重载了括号。一般调用在类中定义的函数的方法是：example_class_instance.func()，
        # 如果只是使用example_class_instance()，那么这个操作就是在调用__call__这个内置方法
        # img_out = net(image)实际上就是调用了net的__call__方法，net的__call__方法没有显式定义，
        # 那么就使用它的父类方法，也就是调用nn.Module的__call__方法，它调用了forward方法，又有，net类中定义了forward方法，所以使用重写的forward方法
        loss = loss_func(img_out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        summary.add_scalar('bceloss', loss, i) # 前三个参数：tag：要求是一个string，用以描述该标量数据图的标题
        # scalar_value ：可以简单理解为一个y轴值的列表 global_step：可以简单理解为一个x轴值的列表，与y轴的值相对应
    torch.save(net.state_dict(), r'SAVE/Unet.pt')
    img, label = data[90]
    img = torch.unsqueeze(img, dim=0).cuda()
    net.eval()
    out = net(img)
    save_image(out, 'Log_imgs/segimg_ep{}_90th_pic.jpg'.format(epoch, i), nrow=1, scale_each=True)
    print('finish {} epoch'.format(epoch))
