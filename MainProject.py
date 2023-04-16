from __future__ import print_function
import copy
import multiprocessing

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import json

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # inherit the module
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 将一个图片的out_channels输出为32 卷积核为3*1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # out_channels 输出为64 卷积核3*1
        self.dropout1 = nn.Dropout(0.25)  # 随机省略，防止过拟合
        self.dropout2 = nn.Dropout(0.5)  # 随机省略，防止过拟合
        self.fc1 = nn.Linear(9216, 128)  # size of each input sample and output sample
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)  # 先用第一种方式卷一下
        x = F.relu(x)  # 线性整流函数（激活函数）
        x = self.conv2(x)  # 再卷再激活
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 最大值池化
        x = self.dropout1(x)  # 第一种丢
        x = torch.flatten(x, 1)  # 转化成一维tensor
        x = self.fc1(x)  # 线性化
        x = F.relu(x)  # 激活
        x = self.dropout2(x)  # 第二种丢
        x = self.fc2(x)  # 激活
        output = F.log_softmax(x, dim=1)  # 放入一个tensor 进行log_softmax操作
        return output  # 输出， 这个过程叫做 forward


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch 第几次
    :return:
    """
    model.train()  # 模型训练
    # 这个for  是对 dataloader 中的batch_idx 和 data , target作为元素遍历
    correct = 0.0
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):  # enumerate 将一个可以遍历的东西串联成一个索引序列，同时列出数据和数据下标
        data, target = data.to(device), target.to(device)  # 将数据喂入设备 data 是 input target 是 labels
        optimizer.zero_grad()  # 梯度清零 每一步都要初始化为0
        output = model(data)  # 前向传播 得到训练结果
        loss = F.nll_loss(output, target)  # 计算损失函数
        total_loss += loss.item()
        predicted = torch.max(output, 1)[1]
        correct += (predicted == target).sum().item()
        loss.backward()  # 反向传播
        optimizer.step()  # 更新可训练权重

    correct = correct / len(train_loader.dataset)
    total_loss = total_loss / len(train_loader.dataset)
    training_acc, training_loss = correct, total_loss  # replace this line
    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0.0  # 测试损失
    correct = 0.0  # 正确率
    with torch.no_grad():
        for data, target in test_loader:
            '''Fill your code'''
            # 部署到device上
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = test_loss + F.cross_entropy(output, target).item()
            # 找到概率值最大的索引
            pred = output.max(1, keepdim=True)[1]
            correct = correct + pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        correct = correct / len(test_loader.dataset)
    testing_acc, testing_loss = correct, test_loss  # replace this line
    return testing_acc, testing_loss


def plot(epoches, performance, name):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    epoches = numpy.array(epoches)

    performance = numpy.array(performance)
    plt.plot(epoches, performance)
    plt.xlabel('epoch')
    plt.ylabel(f'{name}')
    plt.title(name)
    plt.savefig(f'images/{name}.png')
    plt.show()


def run(config):
    run_count = config.run_count
    outputName = "output" + str(run_count) + ".txt"
    # 超参数配置

    start = "------------------------------This is the " + str(run_count) + " time------------------------------"
    with open(f'{outputName}', 'w') as txt:
        txt.write(start)
        txt.write("\n")
    print(start)
    use_cuda = (not True) and torch.cuda.is_available()  # test whether cuda is av
    use_mps = (not True) and torch.backends.mps.is_available()  # test whether mps is av

    torch.manual_seed(config.seed)  # set the seed to generate random numbers

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([  # 对输入的图片做一些变换
        transforms.ToTensor(),  # 转化为tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 正则化 -- 降低模型复杂度
    ])
    # 下载数据集
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, shuffle=True, generator=generator)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs, shuffle=True, generator=generator)

    model = Net().to(device)  # 部署到设备上
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)  # 调用模型参数的优化器 使更准确

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 16):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        train_info = {'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss}

        with open(f'{outputName}', 'a') as txt:

            json.dump(train_info, txt)
            txt.write("\n")
        print(train_info)
        test_acc, test_loss = test(model, device, test_loader)
        test_info = {'epoch': epoch, 'test_acc': test_acc, 'test_loss': test_loss}
        with open(f'{outputName}', 'a') as txt:
            json.dump(test_info, txt)
            txt.write("\n")
        print(test_info)
        scheduler.step()
        """update the records, Fill your code"""
        epoches.append(epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
    """plotting training performance with the records"""
    plot(epoches, training_loss, "training_loss" + f"{str(run_count)}")
    plot(epoches, training_accuracies, "training_accuracies" + f"{str(run_count)}")
    """plotting testing performance with the records"""
    plot(epoches, testing_loss, "testing_loss" + f"{str(run_count)}")
    plot(epoches, testing_accuracies, "testing_accuracies" + f"{str(run_count)}")

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn_with_run_count_" + str(run_count) + ".pt")
    return training_loss, training_accuracies, testing_loss, testing_accuracies


def plot_mean(training_acc_results, training_loss_results, testing_acc_results, testing_loss_results):
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    """fill your code"""

    training_acc_results = np.mean(training_acc_results, axis=0)
    training_loss_results = np.mean(training_loss_results, axis=0)
    testing_acc_results = np.mean(testing_acc_results, axis=0)
    testing_loss_results = np.mean(testing_loss_results, axis=0)
    epoches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    epoches = numpy.array(epoches)
    plot(epoches, training_acc_results, "mean_training_acc")
    plot(epoches, training_loss_results, "mean_training_loss")
    plot(epoches, testing_acc_results, "mean_testing_acc")
    plot(epoches, testing_loss_results, "mean_testing_loss")


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=3)

    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    config2 = copy.deepcopy(config)
    config2.seed = 321
    config2.run_count = 2
    config3 = copy.deepcopy(config)
    config3.seed = 666
    config3.run_count = 3
    configs = [config, config2, config3]
    outputsTL = []
    outputsTA = []
    outputsTEL = []
    outputsTEA = []
    results = []
    for i in configs:
        result = pool.apply_async(run, args=(i, ))
        results.append(result)
    for result in results:
        outputTL, outputTA, outputTEL, outputTEA = result.get()
        outputsTL.append(outputTL)
        outputsTA.append(outputTA)
        outputsTEL.append(outputTEL)
        outputsTEA.append(outputTEA)
    pool.close()
    pool.join()

    """plot the mean results"""
    plot_mean(outputsTA, outputsTL, outputsTEA, outputsTEL)
