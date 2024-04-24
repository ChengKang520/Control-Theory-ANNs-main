import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import network
import torch.utils.tensorboard

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from util import set_seed, accuracy
import argparse
import random
from pid import PIDOptimizer
from lpfsgd import LPFSGDOptimizer
from hpfsgd import HPFSGDOptimizer
from fuzzypid import FUZZYPIDOptimizer
from logger import Logger

I=3
I = float(I)
D = 100
D = float(D)


def norm_y(y_one_hot: torch.Tensor):
    return y_one_hot.sub(0.1307).div(0.3081)

@torch.no_grad()
def test(network_ff, linear_cf, test_loader, opts):
    all_outputs = []
    all_labels = []
    all_logits = []

    for (x_test, y_test) in test_loader:
        x_test, y_test = x_test.to(opts.device), y_test.to(opts.device)
        x_test = x_test.view(x_test.shape[0], -1)

        acts_for_labels = []

        # slow method
        for label in range(10):
            test_label = torch.ones_like(y_test).fill_(label)
            test_label = norm_y(F.one_hot(test_label, num_classes=10))
            x_with_labels = torch.cat((x_test, test_label), dim=1)
            
            acts = network_ff(x_with_labels)
            acts = acts.norm(dim=-1)
            acts_for_labels.append(acts)
        
        # these are logits
        acts_for_labels = torch.stack(acts_for_labels, dim=1) #should be BSZxLABELSxLAYERS (10)
        all_outputs.append(acts_for_labels)
        all_labels.append(y_test)

        # quick method
        neutral_label = norm_y(torch.full((x_test.shape[0], 10), 0.1, device=opts.device))
        acts = network_ff(torch.cat((x_test, neutral_label), dim=1))
        logits = linear_cf(acts.view(acts.shape[0], -1))
        all_logits.append(logits)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    slow_acc = accuracy(all_outputs.mean(dim=-1), all_labels, topk=(1,))[0]
    fast_acc = accuracy(all_logits, all_labels, topk=(1,))[0]
    return slow_acc, fast_acc

def train(network_ff, optimizer, linear_cf, optimizer_cf, train_loader, start_block, opts):
    running_loss = 0.
    running_ce = 0.
    balanced_pos_neg = False

    x_pos, y_pos, x_neg, y_neg, x_rand, y_rand = [], [], [], [], [], []
    for (x, y_pos) in train_loader:
        x, y_pos = x.to(opts.device), y_pos.to(opts.device)
        x = x.view(opts.batch_size, -1)

        # positive pairs #
        y_pos_one_hot = norm_y(F.one_hot(y_pos, num_classes=10))
        x_pos = torch.cat((x, y_pos_one_hot), dim=1)

        # negative pairs from random labels #
        y_neg = []
        for i in range(len(y_pos)):
            index_neg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            index_neg.remove(y_pos[0])
            y_neg.append(random.sample(index_neg, 1))
        y_neg = torch.squeeze(torch.as_tensor(y_neg))
        y_neg_one_hot = norm_y(F.one_hot(y_neg, num_classes=10)).to(opts.device)
        x_neg = torch.cat((x, y_neg_one_hot), dim=1)

        # absolutely random labels #
        y_rand = torch.randint(0, 10, (opts.batch_size,), device=opts.device)
        y_rand_one_hot = norm_y(F.one_hot(y_rand, num_classes=10)).to(opts.device)
        x_rand = torch.cat((x, y_rand_one_hot), dim=1)

        # x_neg = x_rand
        # if opts.hard_negatives:
        #     x_neg = torch.cat((x_neg, x_hard), dim=0)
        if (opts.portion_pos + opts.portion_neg) == 1.0:
            balanced_pos_neg = True
        else:
            balanced_pos_neg = False


        with torch.enable_grad():

            if balanced_pos_neg == False:
                z_pos = network_ff(x_pos[:int(opts.portion_pos*opts.batch_size*2),:], cat=False)
                z_neg = network_ff(x_neg[:int(opts.portion_neg*opts.batch_size*2),:], cat=False)
                z_ran = network_ff(x_rand[:int((1.0-opts.portion_pos-opts.portion_neg)*opts.batch_size*2),:], cat=False)

                for idx, (zp, zn, zr) in enumerate(zip(z_pos, z_neg, z_ran)):
                    if idx < start_block:
                        continue

                    positive_loss = torch.log(1 + torch.exp((-zp.norm(dim=-1) + opts.theta))).mean()
                    negative_loss = torch.log(1 + torch.exp((zn.norm(dim=-1) - opts.theta))).mean()
                    random_loss = torch.log(1 + torch.exp((zr.norm(dim=-1)))).mean()
                    loss = positive_loss + negative_loss + random_loss
                    loss.backward()
                    running_loss += loss.detach()
                    optimizer[idx].step()
                    optimizer[idx].zero_grad()

            else:
                z_pos = network_ff(x_pos, cat=False)
                z_neg = network_ff(x_neg, cat=False)

                for idx, (zp, zn) in enumerate(zip(z_pos, z_neg)):
                    if idx < start_block:
                        continue

                    positive_loss = torch.log(1 + torch.exp((-zp.norm(dim=-1) + opts.theta))).mean()
                    negative_loss = torch.log(1 + torch.exp((zn.norm(dim=-1) - opts.theta))).mean()
                    loss = positive_loss + negative_loss
                    loss.backward()
                    running_loss += loss.detach()
                    optimizer[idx].step()
                    optimizer[idx].zero_grad()

    
    running_loss /= len(train_loader)
    running_ce /= len(train_loader)

    return running_loss, running_ce

def main(opts):
    set_seed(opts.seed)

    T_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(MNIST("~/data", train=True, download=True, transform=T_train), 
                              batch_size=opts.batch_size, shuffle=True, drop_last=True, num_workers=8,
                              persistent_workers=True)
    
    test_loader = DataLoader(MNIST("~/data", train=False, download=True, transform=T_test), 
                             batch_size=opts.batch_size, shuffle=True, num_workers=8,
                             persistent_workers=True)

    size = opts.layer_size
    network_ff = network.Network(dims=[28*28 + 10, size, size, size, size]).to(opts.device)
    # print(network_ff)

    # Create one optimizer for evey relu layer (block)
    if opts.optimizer == 'adam':
        optimizers = [
            torch.optim.Adam(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
                for block in network_ff.blocks.children()
        ]
    elif opts.optimizer == 'sgd':
        optimizers = [
            torch.optim.SGD(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
                for block in network_ff.blocks.children()
        ]

    elif opts.optimizer == 'sgdm':
        optimizers = [
            torch.optim.SGD(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, momentum=0.9)
                for block in network_ff.blocks.children()
        ]

    elif opts.optimizer == 'pid':
        optimizers = [
            PIDOptimizer(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, momentum=0.9, I=I, D=D)
                for block in network_ff.blocks.children()
        ]

    elif opts.optimizer == 'lpfsgd':
        optimizers = [
            LPFSGDOptimizer(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, lpf_sgd=True)
                for block in network_ff.blocks.children()
        ]

    elif opts.optimizer == 'hpfsgd':
        optimizers = [
            HPFSGDOptimizer(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, hpf_sgd=True)
            for block in network_ff.blocks.children()
        ]

    elif opts.optimizer == 'fuzzypid':
        optimizers = [
            FUZZYPIDOptimizer(block.parameters(), lr=opts.lr, weight_decay=args.weight_decay, momentum=0.9, I_pid=I, D_pid=D)
            for block in network_ff.blocks.children()
        ]
        # optimizers = [
        #     torch.optim.SGD(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, momentum=0.9)
        #         for block in network_ff.blocks.children()
        # ]

    # Softmax layer for predicting classes from embeddings (fast method)
    linear_cf = nn.Linear(size*network_ff.n_blocks, 10).to(opts.device)
    optimizer_cf = torch.optim.Adam(linear_cf.parameters(), lr=0.0001)

    writer = SummaryWriter()


    plot = Logger('plot_' + opts.optimizer  + '_P' + str(opts.portion_pos) + '_N' + str(opts.portion_neg) + '_Th' + str(opts.theta) + '.txt', title='mnist')
    plot.set_names(['Step', 'Runing Loss', 'Runing CE', 'Train Fast ACC', 'Train Slow ACC', 'Test Fast ACC', 'Test Slow ACC'])


    start_block = 0
    for step in range(1, opts.epochs+1):
        running_loss, running_ce = train(network_ff, optimizers, linear_cf, optimizer_cf,
                                         train_loader, start_block, opts)
        if step % opts.steps_per_block == 0:
            if start_block+1 < network_ff.n_blocks:
                start_block += 1
                print("Freezing block", start_block-1)

        writer.add_scalar("train/loss", running_loss, step)
        writer.add_scalar("train/ce", running_ce, step)

        train_slow_acc, train_fast_acc = test(network_ff, linear_cf, train_loader, opts)
        test_slow_acc, test_fast_acc = test(network_ff, linear_cf, test_loader, opts)

        writer.add_scalar("acc_fast/train", train_fast_acc, step)
        writer.add_scalar("acc_fast/test", test_fast_acc, step)
        writer.add_scalar("acc_slow/train", train_slow_acc, step)
        writer.add_scalar("acc_slow/test", test_slow_acc, step)

        plot.append([step, running_loss, running_ce, train_fast_acc, train_slow_acc, test_fast_acc, test_slow_acc])
        print(f"Step {step:03d} Loss: {running_loss:.4f} CE: {running_ce:.4f}",
              f"-- TRAIN: fast {train_fast_acc:.2f} (err {(100. - train_fast_acc):.2f}) slow {train_slow_acc:.2f} (err {(100. - train_slow_acc):.2f})",
              f"-- TEST: fast {test_fast_acc:.2f} (err {(100. - test_fast_acc):.2f}) - slow {test_slow_acc:.2f} (err {(100. - test_slow_acc):.2f})")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_block', type=int, default=50)
    parser.add_argument('--theta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')  #  cuda  cpu
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--portion_pos', type=float, default=0.5)
    parser.add_argument('--portion_neg', type=float, default=0.5)
    args = parser.parse_args()


    class Opts:
        # hard_negatives = False
        # layer_size = 2000
        # batch_size = 200
        # lr = 0.0001
        # weight_decay = 0
        # epochs = 100
        # steps_per_block = 60
        # theta = 10.
        # seed = 0
        # device = 'cuda'  # 'cpu'    'cuda'

        layer_size = args.layer_size
        batch_size = args.batch_size
        lr = args.lr
        weight_decay = args.weight_decay
        epochs = args.epochs
        steps_per_block = args.steps_per_block
        theta = args.theta
        seed = args.seed
        device = args.device
        optimizer = args.optimizer
        portion_pos = args.portion_pos
        portion_neg = args.portion_neg

    opts = Opts()

    for i_fold in range(1):
        print('***********************************************************************************')
        print(opts)
        main(opts)
        print('***********************************************************************************')


