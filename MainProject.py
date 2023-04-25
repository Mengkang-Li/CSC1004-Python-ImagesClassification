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
        # it's given to write a net function
        super(Net, self).__init__()  # inherit the module
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # out_channel 32 kernel 3*1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # out_channels 64 kernel3*1
        self.dropout1 = nn.Dropout(0.25)  # random dropout
        self.dropout2 = nn.Dropout(0.5)  # random dropout to avoid overfitting
        self.fc1 = nn.Linear(9216, 128)  # size of each input sample and output sample
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()  # it's the train model
    #
    correct = 0.0
    total_loss = 0.0
    # in the for loop, use correct as the number of all the correctly predicted samples, use loss as the sum of all loss
    # Each loss and Each correct are used to record the single loss and correct in a single batch
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        EachLoss = loss.item()
        predicted = torch.max(output, 1)[1]
        correct += (predicted == target).sum().item()
        EachCorrect = (predicted == target).sum().item()
        EachCorrect = EachCorrect / train_loader.batch_size
        loss.backward()
        optimizer.step()
        # output the result of the batch
        outputName = "output" + str(args.run_count) + ".txt"
        info = "TRAIN---: " + str(args.run_count) + "---epoch:" + str(epoch) + "---batch_idx: " + str(batch_idx) + \
               "---Single loss=" + str(EachLoss) + ", ---Single correct=" + str(EachCorrect)
        # write them into the corresponding file
        with open(f'{outputName}', 'a') as txt:
            txt.write(info)
            txt.write("\n")

        print(info)

    # to get the average correct ratio for the whole dataset, so it's using the sum of the loss and correct of the whole
    # dataset
    correct = correct / len(train_loader.dataset)
    total_loss = total_loss / len(train_loader.dataset)
    training_acc, training_loss = correct, total_loss  # replace this line
    return training_acc, training_loss


def test(args, model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0.0
    correct = 0.0
    # use similar structure of the train method
    with torch.no_grad():
        # Each loss and correct are used to record the accuracy or loss in a single batch
        # predicted is used to find the predicted label, so that it can be compared with the real target and then get
        # the correct
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            EachLoss = F.cross_entropy(output, target).item()
            test_loss = test_loss + F.cross_entropy(output, target).item()
            predicted = output.max(1, keepdim=True)[1]
            EachCorrect = predicted.eq(target.view_as(predicted)).sum().item()
            correct = correct + predicted.eq(target.view_as(predicted)).sum().item()
            EachCorrect = EachCorrect / test_loader.batch_size
            outputName = "output" + str(args.run_count) + ".txt"
            info = "TEST---: " + str(args.run_count) + "---test batch: " + str(batch_idx) + "---Single loss=" + \
                str(EachLoss) + ", ---Single correct=" + str(EachCorrect)
            # write the data to the corresponding file
            with open(f'{outputName}', 'a') as txt:
                txt.write(info)
                txt.write("\n")
            print(info)
        # similarly, use the total loss and correct to compute the final average results
        test_loss /= len(test_loader.dataset)
        correct = correct / len(test_loader.dataset)
    testing_acc, testing_loss = correct, test_loss
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
    # set the fig size to show all the information in the graph
    plt.figure(figsize=(8, 6))
    plt.plot(epoches, performance)  # plot the graph
    plt.xlabel('epoch')  # set the x label
    plt.ylabel(f'{name}')  # set the y label
    plt.title(f'{name}' + " with Epochs", fontsize=20)  # set the title and enlarge the font size
    plt.savefig(f'images/{name}.png')  # restore the image
    plt.show()  # show and clear the image


def run(config):
    # use the run_count to label what is the sequence number of the run
    run_count = config.run_count
    # set the name of the output file using run_count
    outputName = "output" + str(run_count) + ".txt"
    # 超参数配置

    start = "------------------------------This is the " + str(run_count) + " time------------------------------"
    # in the terminal, show the process is operating
    # start writing the output file
    with open(f'{outputName}', 'w') as txt:
        txt.write(start)
        txt.write("\n")
    print(start)
    use_cuda = (not True) and torch.cuda.is_available()  # test whether cuda is av
    use_mps = (not True) and torch.backends.mps.is_available()  # test whether mps is av

    torch.manual_seed(config.seed)  # set the seed to generate random numbers

    # choose the device
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # add the config of the dataloader, parameter 'generator' is used to set the random seed
    # of the data_loader's shuffle process
    generator = torch.Generator().manual_seed(config.seed)
    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True, 'generator': generator}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True, 'generator': generator}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([  # transform the data
        transforms.ToTensor(),  # transform to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # normalize
    ])
    # download the dataset
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    # add the config to the dataloader
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # deploy the program to the device
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)  # optimizer makes it more accurate

    # create these lists to memorize the result of all epochs
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    # in the loop, use train_info to show the information in the training test_info in the testing process
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        train_info = {'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss}
        # write into the corresponding file
        with open(f'{outputName}', 'a') as txt:
            json.dump(train_info, txt)
            txt.write("\n")
        print(train_info)
        test_acc, test_loss = test(config, model, device, test_loader)
        test_info = {'epoch': epoch, 'test_acc': test_acc, 'test_loss': test_loss}
        with open(f'{outputName}', 'a') as txt:
            json.dump(test_info, txt)
            txt.write("\n")
        print(test_info)
        scheduler.step()
        # update the info in the list
        epoches.append(epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)

    # plot the 4 graphs
    plot(epoches, training_loss, "training_loss" + f"{str(run_count)}")
    plot(epoches, training_accuracies, "training_accuracies" + f"{str(run_count)}")
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
    # get the mean value of the results of three runs
    training_acc_results = np.mean(training_acc_results, axis=0)
    training_loss_results = np.mean(training_loss_results, axis=0)
    testing_acc_results = np.mean(testing_acc_results, axis=0)
    testing_loss_results = np.mean(testing_loss_results, axis=0)
    # build the epochs array
    epoches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    epoches = numpy.array(epoches)
    # plot the four images
    plot(epoches, training_acc_results, "mean_training_acc")
    plot(epoches, training_loss_results, "mean_training_loss")
    plot(epoches, testing_acc_results, "mean_testing_acc")
    plot(epoches, testing_loss_results, "mean_testing_loss")


if __name__ == '__main__':
    # it is said that print does not need a lock, and in the file there's not any situation where different processes
    # write in the same file, since there's three file for three processes
    # just for plot we need a lock to achieve synchronization
    # create a multiprocessing pool
    pool = multiprocessing.Pool(processes=3)

    arg = read_args()
    # load the config
    config = load_config(arg)

    # create three config dictionaries, which differs in run_count and seed
    config2 = copy.deepcopy(config)
    config2.seed = 321
    config2.run_count = 2
    config3 = copy.deepcopy(config)
    config3.seed = 666
    config3.run_count = 3
    configs = [config, config2, config3]
    # create four lists to record the results of three runs, so that I can compute the mean results in the plot_mean
    outputsTL = []
    outputsTA = []
    outputsTEL = []
    outputsTEA = []
    results = []
    # hand out the configs with three process runs
    for i in configs:
        result = pool.apply_async(run, args=(i,))
        results.append(result)
    # record the results
    for result in results:
        outputTL, outputTA, outputTEL, outputTEA = result.get()
        outputsTL.append(outputTL)
        outputsTA.append(outputTA)
        outputsTEL.append(outputTEL)
        outputsTEA.append(outputTEA)
    # close the pool
    pool.close()
    pool.join()

    # plot the mean result
    plot_mean(outputsTA, outputsTL, outputsTEA, outputsTEL)
