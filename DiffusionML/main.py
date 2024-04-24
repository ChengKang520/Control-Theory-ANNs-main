# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from logger import Logger
from torchvision.utils import save_image
import sys
import os
import argparse
from fuzzypid import FUZZYPIDOptimizer
from pid import PIDOptimizer
from lpfsgd import LPFSGDOptimizer
from hpfsgd import HPFSGDOptimizer
import numpy as np


I_pid = 3
I_pid = float(I_pid)
D_pid = 100
D_pid = float(D_pid)


# MNIST Dataset
mean = np.array([0.5])
std = np.array([0.5])

transform = transforms.Compose([transforms.Resize([28, 28]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))




# build network
z_dim = 100
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
# D = Discriminator(mnist_dim).to(device)

# loss
criterion = nn.BCELoss()


def D_train(x, D_optimizer):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x, G_optimizer):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='no-rep', help='Choose Optimizers: SGD, SGD-M, Adam, PID')
    parser.add_argument('--controller_type', type=str, default='pid', help='sgd, sgdm, adam, pid')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='')
    parser.add_argument('--samples_path', type=str, default='./samples/adam/', help='')
    parser.add_argument('--model_path', type=str, default='./models/adam/', help='')
    parser.add_argument('--bsz', type=int, default=10, help='')
    parser.add_argument('--n_epoch', type=int, default=200, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')

    args = parser.parse_args()
    print('**************************  args  ******************************')
    print(f"arg is: {args}")
    print('**************************  args  ******************************')


    n_epoch = args.n_epoch
    bs = args.bsz

    # Directories for storing model and output samples
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    samples_path = args.samples_path
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)


    if args.controller_type == 'sgd':
        G_optimizer = optim.SGD(G.parameters(), lr=args.learning_rate)
        D_optimizer = optim.SGD(D.parameters(), lr=args.learning_rate)
    elif args.controller_type == 'sgdm':
        G_optimizer = optim.SGD(G.parameters(), lr=args.learning_rate, momentum=0.9)
        D_optimizer = optim.SGD(D.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.controller_type == 'adam':
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate)
        D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate)
    elif args.controller_type == 'pid':
        G_optimizer = PIDOptimizer(G.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)
        D_optimizer = PIDOptimizer(D.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)
    elif args.controller_type == 'lpfsgd':
        G_optimizer = LPFSGDOptimizer(G.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, lpf_sgd=True)
        D_optimizer = LPFSGDOptimizer(D.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, lpf_sgd=True)
    elif args.controller_type == 'hpfsgd':
        G_optimizer = HPFSGDOptimizer(G.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, hpf_sgd=True)
        D_optimizer = HPFSGDOptimizer(D.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, hpf_sgd=True)
    elif args.controller_type == 'fuzzypid':
        G_optimizer = FUZZYPIDOptimizer(G.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)
        D_optimizer = FUZZYPIDOptimizer(D.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
    plot.set_names(['loss_d', 'loss_g'])

    for epoch in range(1, n_epoch+1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(D_train(x, D_optimizer))
            G_losses.append(G_train(x, G_optimizer))

        plot.append([torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))])
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        with torch.no_grad():
            test_z = Variable(torch.randn(bs, z_dim).to(device))
            generated = G(test_z)
            save_image(generated.view(generated.size(0), 1, 28, 28), samples_path + '/sample_' + str(epoch) + '.png')

    with torch.no_grad():
        test_z = Variable(torch.randn(bs, z_dim).to(device))
        generated = G(test_z)
        save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_' + args.controller_type + '.png')

        torch.save(D.state_dict(), os.path.join(model_path, 'D_gen.pkl'))
        torch.save(G.state_dict(), os.path.join(model_path, 'G_gen.pkl'))

    plot.close()





