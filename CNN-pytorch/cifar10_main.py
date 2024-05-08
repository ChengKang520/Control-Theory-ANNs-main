'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from fuzzypid import FUZZYPIDOptimizer
from pid import PIDOptimizer
from lpfsgd import LPFSGDOptimizer
from hpfsgd import HPFSGDOptimizer
import torchvision
import torchvision.transforms as transforms
from model_input import resnet, vgg, densenet, mobilenetv2, efficientnet
import torchvision.datasets as datasets
import torch.utils.data as data
import os
import argparse
import torchvision.models as models
from models import *
# from helps import progress_bar


I = 5
I = float(I)
D = 100
D = float(D)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # parser.add_argument('--experiment', type=str, default='no-rep', help='Choose Optimizers: SGD, SGD-M, Adam, PID')
    parser.add_argument('--model_type', type=str, default='sgd', help='')
    parser.add_argument('--controller_type', type=str, default='sgd', help='sgd, sgdm, adam, pid')
    parser.add_argument('--model_path', type=str, default=None, help='')
    parser.add_argument('--num_classes', type=int, default=10, help='')
    parser.add_argument('--bsz', type=int, default=100, help='')
    parser.add_argument('--n_epoch', type=int, default=20, help='')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='')
    parser.add_argument('--experiment', type=str, default='None', help='Choose Learning Rate Decay Method: LinearLR, CosineAnnealingLR, ExponentialLR')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    args = parser.parse_args()
    print('**************************  args  ******************************')
    print(f"arg is: {args}")
    print('**************************  args  ******************************')

    num_epochs = args.n_epoch
    batch_size = args.bsz
    learning_rate = args.learning_rate

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    if args.num_classes == 10:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)
    elif args.num_classes == 100:
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)
    elif args.num_classes == 200:
        data_dir = './data/tiny-224/'
        num_workers = {'train': 2, 'val': 0, 'test': 0}
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        trainloader = data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers['train'])
        testloader = data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=num_workers['test'])
        train_sizes = len(image_datasets['train'])
        test_sizes = len(image_datasets['test'])

        # dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
        #                for x in ['train', 'val', 'test']}
        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    parent_dir = "/home/kangchen/Controllable_AI/pytorch-mnist-CNN/"
    model_save_path = os.path.join(parent_dir, args.model_path, args.model_type)

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    # Model
    print('==> Building model..')


    if args.num_classes == 200:

        if args.model_type == 'vgg19':
            net = models.vgg19(pretrained=False)
            net.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            net.classifier.out_features = 200

        elif args.model_type == 'resnet18':
            net = models.resnet18(pretrained=False)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.fc.out_features = 200

        elif args.model_type == 'resnet50':
            net = models.resnet50(pretrained=False)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.fc.out_features = 200
            net = net.to(device)

        elif args.model_type == 'resnet101':
            net = models.resnet101(pretrained=False)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.fc.out_features = 200
            net = net.to(device)

        elif args.model_type == 'mobilenetv2':
            net = models.mobilenet_v2(pretrained=False)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.classifier.out_features = 200
            net = net.to(device)

        elif args.model_type == 'efficientnet':
            net = models.efficientnet_b0(pretrained=False)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.classifier.out_features = 200
            net = net.to(device)

        elif args.model_type == 'densenet121':
            net = models.densenet121(pretrained=False)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.classifier.out_features = 200
            net = net.to(device)
        net = net.to(device)

    elif (args.num_classes == 10) or (args.num_classes == 100):

        if args.model_type == 'vgg19':
            net = vgg.VGG('VGG19', num_classes=args.num_classes)
        if args.model_type == 'resnet18':
            net = resnet.ResNet18(num_classes=args.num_classes)
        elif args.model_type == 'resnet50':
            net = resnet.ResNet50(num_classes=args.num_classes)
        elif args.model_type == 'resnet101':
            net = resnet.ResNet101(num_classes=args.num_classes)
        elif args.model_type == 'mobilenetv2':
            net = mobilenetv2.MobileNetV2(num_classes=args.num_classes)
        elif args.model_type == 'efficientnet':
            net = efficientnet.EfficientNetB0(num_classes=args.num_classes)
        elif args.model_type == 'densenet121':
            net = densenet.DenseNet121(num_classes=args.num_classes)



    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    if args.controller_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    elif args.controller_type == 'sgdm':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    elif args.controller_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    elif args.controller_type == 'pid':
        optimizer = PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9, I=I,
                                 D=D)

    elif args.controller_type == 'lpfsgd':
        optimizer = LPFSGDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, lpf_sgd=True)

    elif args.controller_type == 'hpfsgd':
        optimizer = HPFSGDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, hpf_sgd=True)

    elif args.controller_type == 'fuzzypid':
        optimizer = FUZZYPIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9,
                                      I_pid=I,
                                      D_pid=D)


    ##########  Learning Rate Decay Setup  ##########
    if args.experiment == 'LinearLR':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=200)
    elif args.experiment == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
    elif args.experiment == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.experiment == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    elif args.experiment == 'None':
        scheduler = optimizer



    if args.num_classes == 200:
        plot_train = Logger('TinyImageNet' + str(args.num_classes) + '_plot_train_' + args.model_type + '_' + args.experiment + '.txt', title='mnist')
        plot_train.set_names(['Epoch', 'Step', 'Loss', 'Acc'])
        plot_test = Logger('TinyImageNet' + str(args.num_classes) + '_plot_test_' + args.model_type + '_' + args.experiment + '.txt', title='mnist')
        plot_test.set_names(['Epoch', 'Loss', 'Acc'])

    elif (args.num_classes == 10) or (args.num_classes == 100):
        plot_train = Logger('CIFAR' + str(args.num_classes) + '_plot_train_' + args.model_type + '_' + args.experiment + '.txt', title='mnist')
        plot_train.set_names(['Epoch', 'Step', 'Loss', 'Acc'])
        plot_test = Logger('CIFAR' + str(args.num_classes) + '_plot_test_' + args.model_type + '_' + args.experiment + '.txt', title='mnist')
        plot_test.set_names(['Epoch', 'Loss', 'Acc'])


    acc = []
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    for epoch in range(start_epoch, start_epoch + num_epochs):

        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 50 == 0:
                plot_train.append([epoch + 1, batch_idx + 1, train_loss / (batch_idx + 1), 100. * correct / total])
                # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # if (batch_idx + 1) % 50 == 0:
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        plot_test.append([epoch + 1, test_loss / (batch_idx + 1), acc])
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            torch.save(state, model_save_path + '/ckpt.pth')
            best_acc = acc

        scheduler.step()


