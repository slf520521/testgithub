import torch
import Config
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from sklearn import metrics
import math

batch_size = 256  # 一批数据有多少条
input_size = 63  # 输入的维度
hidden_size = 32  # GRU隐藏层的维度
workers = 2  # 用几个进程加载数据
learning_rate = 1e-2  # 学习率，学习率越高梯度下降的越快
epochs = 50  # 总共训练多少轮
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 数据放到gpu上还是cpu上
datatype = "mimic3"  # 数据集是mimic3
save_model_dir = '/home/lb/Model/GRUmodel_' + datatype + '.pth'  # 模型保存的路径
testepoch = 10


def get_data():
    # 读取数据集，这里输入数据里面的缺失值都是0
    x_torch = pickle.load(open('dataset/lb_' + datatype + '_x_for_missingvalue.p', 'rb'))
    y_torch = pickle.load(open('dataset/lb_' + datatype + '_y.p', 'rb'))

    print(x_torch.shape)
    print(y_torch.shape)

    # 划分训练集验证集和测试集，8：1：1
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    N = len(x_torch)

    training_x = x_torch[: int(train_ratio * N)]
    validing_x = x_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_x = x_torch[int((train_ratio + valid_ratio) * N):]

    training_y = y_torch[: int(train_ratio * N)]
    validing_y = y_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_y = y_torch[int((train_ratio + valid_ratio) * N):]

    train_deal_dataset = TensorDataset(training_x, training_y)
    test_deal_dataset = TensorDataset(testing_x, testing_y)
    valid_deal_dataset = TensorDataset(validing_x, validing_y)

    train_loader = DataLoader(dataset=train_deal_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=workers)

    test_loader = DataLoader(dataset=test_deal_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=workers)

    valid_loader = DataLoader(dataset=valid_deal_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=workers)

    return train_loader, test_loader, valid_loader


# 这是官方的GRU
class GRUofficial(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUofficial, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 指定输入维度，隐藏层维度，GRU层数，是否把batchsize放到第一个维度，是否是双向RNN
        self.GRUofficialAPI = nn.GRU(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     batch_first=True,
                                     bidirectional=False)

        self.outlinear = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )  # 输出层

    def forward(self, input):
        # 把数据放到gpu上，转成float类型
        input = input.to(device).float()  # batchsize,seqlen,inputsize

        out, h = self.GRUofficialAPI(input)
        '''
            out第一个维度是几个样本，batch
            第二个维度是几个时间步，一个时间步出一个h
            第三个维度是每个hstate的维数，就是numdirections*hiddensize
            说白了out就是把每次rnncell输出的hstate打包在了一起
            out[0][0]是第一个样本第一个时间步的rnncell输出的hstate
            out[0][1]是第一个样本第二个时间步的rnncell输出的hstate
            ht的维度[num_layers * num_directions, batch_size, hidden_size],
            如果是单向单层的GRU那么一个样本只有一个hidden，即，ht的维度为[1, batch_size, hidden_size]
        
        '''
        # 因为ht的维度为[1, batch_size, hidden_size]，我们想把那个1去掉，就用squeeze函数
        # 想在某个维度扩充就用unsqueeze,比如h的维度是[A,B],我们想让它变成[A,1,B],在第二个维度扩充，就写h.unsqueeze(dim=1)（dim从0开始算）
        output = self.outlinear(h.squeeze())  # 最后一个单元的输出，接一个带有Sigmoid激活函数的线性层，因为我们的任务是分类任务
        # print(output.shape)  #output矩阵的形状现在是(batchsize,outputsize)

        return output


# 这是自己实现的GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 设置可以训练的参数矩阵
        self.w_xr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_xz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hz = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.w_xh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        self.outlinear = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )  # 输出层

        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input):
        input = input.to(device).float()  # (batchsize,seqlen,inputsize)

        batch_size = input.size(0)  # 一个batch的大小
        step_size = input.size(1)  # 时间步

        # 初始化隐藏状态矩阵h为零矩阵
        h = torch.zeros(batch_size, self.hidden_size).to(device)

        # 这里面存放每一个时间步出来的h
        lisths = []

        # 一个时间步一个时间步的计算
        for i in range(step_size):
            # 取input每个时间步的数据
            x = input[:, i, :]

            # --------------------------------GRU核心公式-----------------------------------
            # x形状是(batchsize,inputsize),w_xz矩阵形状是(inputsize,hiddensize)
            # torch.mm是矩阵乘法，这样(torch.mm(x,self.w_xz)的形状是(batchsize,hiddensize)

            z = torch.sigmoid((torch.mm(x, self.w_xz) + torch.mm(h, self.w_hz) + self.b_z))
            r = torch.sigmoid((torch.mm(x, self.w_xr) + torch.mm(h, self.w_hr) + self.b_r))
            h_tilde = torch.tanh((torch.mm(x, self.w_xh) + torch.mm(r * h, self.w_hh) + self.b_h))
            h = (1 - z) * h + z * h_tilde

            # --------------------------------GRU核心公式-----------------------------------
            # h的形状是(batch_size,hidden_size)

            # 把每个时间步出来的h都存到list里
            lisths.append(h)

        # 用torch.stack把装有tensor的list转为torch.tensor类型,dim=1是指从第二个维度转化，因为seqlen在第二个维度上
        # 所以hs的形状是(batchsize,seqlen,hiddensize)
        hs = torch.stack(lisths, dim=1)  # 全部cell所计算的隐藏状态的集合

        # 此时的h是最后一个时间步计算出来的h，可以用这个作为最后的输出
        output = self.outlinear(h)  # 最后一个单元的输出，接一个带有Sigmoid激活函数的线性层，因为我们的任务是分类任务
        # output矩阵的形状现在是(batchsize,outputsize)

        return output


# model = GRUofficial(input_size, hidden_size,1)
model = GRU(input_size, hidden_size, 1)
# 把模型放到gpu上
model.to(device)
# 指定优化器为adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model, train_loader, valid_loader):
    # 设置loss的数组
    train_loss_array = []
    # 是否提前停止，提前停止的代码在Config里
    # 提前停止就是训练的过程中，如果验证集的loss不再下降，就停止训练了
    Early_stopping = Config.EarlyStopping()

    for epoch in range(epochs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
        # model.train()是保证BN层能够用到每一批数据的均值和方差。
        # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        model.train()
        for i, data in enumerate(train_loader):
            # 这里的i是所有的traindata根据一个batchsize分成的总数
            # i是第i个batchsize

            # 从data中取出输入和标签，然后都放到gpu上
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.float()

            # 前向传播
            out = model(inputs)  # (256,1)
            out = out.to(device).squeeze()  # (256,1)->(256)
            # 二分类任务loss函数使用binary cross entropy，BCE
            lossF = torch.nn.BCELoss(size_average=True).to(device)
            # 得到loss分数
            batch_loss = lossF(out, labels)

            # 反向传播
            optimizer.zero_grad()
            batch_loss.backward(retain_graph=True)
            optimizer.step()
        # 每四个epoch把学习率降为原来的一半，防止步长太大反复横跳无法收敛到最优解
        if epoch % 4 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5

        if (epoch + 1) % 1 == 0:  # 每 1 次输出结果
            print('Epoch: {}, Train Loss: {}'.format(epoch + 1, batch_loss.detach().data))
            train_loss_array.append(batch_loss.detach().data)

            # 每个epoch都在验证集上过一遍
            device = torch.device("cpu")

            # 如果模型中有BN层(Batch Normalization）和Dropout，在测试/验证时添加model.eval()。
            # model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试/验证过程中要保证BN层的均值和方差不变。
            # 对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
            model.eval()
            valid_losses = []
            for i, data in enumerate(valid_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.float()

                # 前向传播
                out = model(inputs)
                out = out.to(device).squeeze()
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                batch_loss = lossF(out, labels)

                # 验证集就没有反向传播了
                valid_losses.append(batch_loss.detach().data)

            valid_loss = np.average(valid_losses)
            print('Epoch: {}, Valid Loss: {}'.format(epoch + 1, valid_loss))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            Early_stopping(valid_loss, model, state, save_model_dir)
            # 如果满足条件就提前停止，我这里设置的条件是经过四个epoch 验证集loss都不下降
            if Early_stopping.early_stop:
                print("Early stopping")
                break


def test_model(model, test_loader):
    device = torch.device("cpu")
    # 切换成验证模式，这个模式下模型参数被固定，不会再更新
    model.eval()
    test_loss_array = []
    # 把模型的输出outs和标签都放到list里，之后计算auroc和auprc要用
    outs = list()
    labelss = list()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.float()

            # print(inputs.shape)
            # 前向传播
            out = model(inputs)

            out = out.to(device).squeeze()
            lossF = torch.nn.BCELoss(size_average=True).to(device)
            batch_loss = lossF(out, labels)

            outs.extend(list(out.numpy()))
            labelss.extend(list(labels.numpy()))

            print('Test loss:{}'.format(float(batch_loss.data)))
            test_loss_array.append(float(batch_loss.data))

    # 转成numpy.array类型
    outs = np.array(outs)
    labelss = np.array(labelss)

    auroc = metrics.roc_auc_score(labelss, outs)

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(labelss, outs)
    auprc = metrics.auc(recalls, precisions)

    return auroc, auprc


def main():
    # 取数据
    train_loader, test_loader, valid_loader = get_data()
    # 训练模型
    train_model(model, train_loader, valid_loader)
    # 加载保存的模型
    checkpoint = torch.load(save_model_dir)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    aurocs = []
    auprcs = []

    for i in range(testepoch):
        train_loader, test_loader, valid_loader = get_data()
        auroc, auprc = test_model(model, test_loader)
        aurocs.append(auroc)
        auprcs.append(auprc)

    auroc_mean = np.mean(aurocs)
    auroc_std = np.std(aurocs, ddof=1)
    auprc_mean = np.mean(auprcs)
    auprc_std = np.std(auprcs, ddof=1)

    print("auroc 平均值为：" + str(auroc_mean) + " 标准差为：" + str(auroc_std))
    print("auprc 平均值为：" + str(auprc_mean) + " 标准差为：" + str(auprc_std))

    return


if __name__ == '__main__':
    main()
