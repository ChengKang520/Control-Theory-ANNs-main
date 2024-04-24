import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from fuzzypid import FUZZYPIDOptimizer
from pid import PIDOptimizer
from lpfsgd import LPFSGDOptimizer
from hpfsgd import HPFSGDOptimizer
from torch import optim
import matplotlib.pyplot as plt
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F
import sys
import argparse
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
input_size = 784
hidden_size = 1000
num_classes = 10

I = 5
I = float(I)
D = 100
D = float(D)

# then whenever you get a new Tensor or Module
# this won't copy if they are already on the desired device

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='no-rep', help='Choose Optimizers: SGD, SGD-M, Adam, PID')
    parser.add_argument('--controller_type', type=str, default='sgd', help='sgd, sgdm, adam, pid')
    parser.add_argument('--model_path', type=str, default=None, help='')
    parser.add_argument('--bsz', type=int, default=100, help='')
    parser.add_argument('--n_epoch', type=int, default=20, help='')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='')

    args = parser.parse_args()
    print('**************************  args  ******************************')
    print(f"arg is: {args}")
    print('**************************  args  ******************************')

    # Directories for storing model and output samples
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    num_epochs = args.n_epoch
    batch_size = args.bsz
    learning_rate = args.learning_rate

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    net = Net(input_size, hidden_size, num_classes).to(device)
    # net.cuda()
    net.train()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    if args.controller_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        logger = Logger('log_' + args.controller_type + '.txt', title='mnist')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
        plot.set_names(['Epoch', 'Step', 'Loss', 'Acc'])

    elif args.controller_type == 'sgdm':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        logger = Logger('log_' + args.controller_type + '.txt', title='mnist')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
        plot.set_names(['Epoch', 'Step', 'Loss', 'Acc'])

    elif args.controller_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        logger = Logger('log_' + args.controller_type + '.txt', title='mnist')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
        plot.set_names(['Epoch', 'Step', 'Loss', 'Acc'])

    elif args.controller_type == 'pid':
        optimizer = PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9, I=I,
                                 D=D)
        logger = Logger('log_' + args.controller_type + '.txt', title='mnist')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
        plot.set_names(['Epoch', 'Step', 'Loss', 'Acc'])

    elif args.controller_type == 'lpfsgd':
        optimizer = LPFSGDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, lpf_sgd=True)
        logger = Logger('log_' + args.controller_type + '.txt', title='mnist')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
        plot.set_names(['Epoch', 'Step', 'Loss', 'Acc'])

    elif args.controller_type == 'hpfsgd':
        optimizer = HPFSGDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, hpf_sgd=True)
        logger = Logger('log_' + args.controller_type + '.txt', title='mnist')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
        plot.set_names(['Epoch', 'Step', 'Loss', 'Acc'])

    elif args.controller_type == 'fuzzypid':
        optimizer = FUZZYPIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9,
                                      I_pid=I,
                                      D_pid=D)
        logger = Logger('log_' + args.controller_type + '.txt', title='mnist')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
        plot.set_names(['Epoch', 'Step', 'Loss', 'Acc'])

    # Train the Model
    for epoch in range(num_epochs):
        train_loss_log = AverageMeter()
        train_acc_log = AverageMeter()
        val_loss_log = AverageMeter()
        val_acc_log = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            # images = Variable(images.view(-1, 28*28).cuda())
            # labels = Variable(labels.cuda())

            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            train_loss_log.update(train_loss.data, images.size(0))
            train_acc_log.update(prec1, images.size(0))

            if (i + 1) % 50 == 0:
                plot.append([epoch + 1, i + 1, train_loss_log.avg, train_acc_log.avg])
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, train_loss_log.avg,
                         train_acc_log.avg))

        # Test the Model
        net.eval()
        correct = 0
        loss = 0
        total = 0
        for images, labels in test_loader:
            # images = Variable(images.view(-1, 28*28)).cuda()
            # labels = Variable(labels).cuda()
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            outputs = net(images)
            test_loss = criterion(outputs, labels)
            val_loss_log.update(test_loss.data, images.size(0))
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            val_acc_log.update(prec1, images.size(0))

        logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
        print('Accuracy of the network on the 10000 test images: %d %%' % (val_acc_log.avg))
        print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))

    torch.save(net.state_dict(), os.path.join(model_path, 'cnn.pkl'))

    logger.close()
    plot.close()

    # logger.cpu().plot()

