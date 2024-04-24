from model import Discriminator, Generator
from data_loader import get_loaders
from utils import generate_imgs
from fuzzypid import FUZZYPIDOptimizer
from pid import PIDOptimizer
from lpfsgd import LPFSGDOptimizer
from hpfsgd import HPFSGDOptimizer
from torch import optim
import numpy as np
import torch
import sys
import os
import argparse
from logger import Logger

I_pid = 3
I_pid = float(I_pid)
D_pid = 100
D_pid = float(D_pid)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='no-rep', help='Choose Optimizers: SGD, SGD-M, Adam, PID')
    parser.add_argument('--controller_type', type=str, default='sgd', help='sgd, sgdm, adam, pid')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='')
    parser.add_argument('--samples_path', type=str, default=None, help='')
    parser.add_argument('--model_path', type=str, default=None, help='')
    parser.add_argument('--bsz', type=int, default=10, help='')
    parser.add_argument('--n_epoch', type=int, default=200, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')

    args = parser.parse_args()
    print('**************************  args  ******************************')
    print(f"arg is: {args}")
    print('**************************  args  ******************************')


    EPOCHS = args.n_epoch  # 50-300
    BATCH_SIZE = args.bsz
    LOAD_MODEL = False

    IMAGE_SIZE = 16
    A_DS = 'usps'
    A_Channels = 1

    B_DS = 'mnist'
    B_Channels = 1

    # Directories for storing model and output samples
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    samples_path = args.samples_path
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)
    db_path = './data'
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    plot = Logger('plot_' + args.controller_type + '.txt', title='mnist')
    plot.set_names(['a_d_loss', 'a_g_loss', 'a_g_ctnt' , 'b_d_loss', 'b_g_loss', 'a_g_ctnt'])


    # Networks
    ab_gen = Generator(in_channels=A_Channels, out_channels=B_Channels)
    ba_gen = Generator(in_channels=B_Channels, out_channels=A_Channels)
    a_disc = Discriminator(channels=A_Channels)
    b_disc = Discriminator(channels=B_Channels)

    # Load previous model
    if LOAD_MODEL:
        ab_gen.load_state_dict(torch.load(os.path.join(model_path, 'ab_gen.pkl')))
        ba_gen.load_state_dict(torch.load(os.path.join(model_path, 'ba_gen.pkl')))
        a_disc.load_state_dict(torch.load(os.path.join(model_path, 'a_disc.pkl')))
        b_disc.load_state_dict(torch.load(os.path.join(model_path, 'b_disc.pkl')))


    if args.controller_type == 'sgd':
        g_opt = optim.SGD(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate)
        d_opt = optim.SGD(list(a_disc.parameters()) + list(b_disc.parameters()), lr=args.learning_rate)
    elif args.controller_type == 'sgdm':
        g_opt = optim.SGD(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, momentum=0.9)
        d_opt = optim.SGD(list(a_disc.parameters()) + list(b_disc.parameters()), lr=args.learning_rate, momentum=0.9)
    elif args.controller_type == 'adam':
        g_opt = optim.Adam(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=2e-5)
        d_opt = optim.Adam(list(a_disc.parameters()) + list(b_disc.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=2e-5)
    elif args.controller_type == 'pid':
        g_opt = PIDOptimizer(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)
        d_opt = PIDOptimizer(list(a_disc.parameters()) + list(b_disc.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)
    elif args.controller_type == 'lpfsgd':
        g_opt = LPFSGDOptimizer(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, lpf_sgd=True)
        d_opt = LPFSGDOptimizer(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, lpf_sgd=True)
    elif args.controller_type == 'hpfsgd':
        g_opt = HPFSGDOptimizer(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, hpf_sgd=True)
        d_opt = HPFSGDOptimizer(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, hpf_sgd=True)
    elif args.controller_type == 'fuzzypid':
        g_opt = FUZZYPIDOptimizer(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)
        d_opt = FUZZYPIDOptimizer(list(a_disc.parameters()) + list(b_disc.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, I_pid=I_pid, D_pid=D_pid)


    # Data loaders
    a_loader, b_loader = get_loaders(db_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, a_ds=A_DS, b_ds=B_DS)
    iters_per_epoch = min(len(a_loader), len(b_loader))

    # Fix images for viz
    a_fixed = next(iter(a_loader))[0]
    b_fixed = next(iter(b_loader))[0]

    # GPU Compatibility
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        ab_gen, ba_gen = ab_gen.cuda(), ba_gen.cuda()
        a_disc, b_disc = a_disc.cuda(), b_disc.cuda()

        a_fixed = a_fixed.cuda()
        b_fixed = b_fixed.cuda()

    # Cycle-GAN Training
    for epoch in range(EPOCHS):
        ab_gen.train()
        ba_gen.train()
        a_disc.train()
        b_disc.train()

        for i, (a_data, b_data) in enumerate(zip(a_loader, b_loader)):

            # Loading data
            a_real, _ = a_data
            b_real, _ = b_data

            if is_cuda:
                a_real, b_real = a_real.cuda(), b_real.cuda()

            # Fake Images
            b_fake = ab_gen(a_real)
            a_fake = ba_gen(b_real)

            # Training discriminator
            a_real_out = a_disc(a_real)
            a_fake_out = a_disc(a_fake.detach())
            a_d_loss = (torch.mean((a_real_out - 1) ** 2) + torch.mean(a_fake_out ** 2)) / 2

            b_real_out = b_disc(b_real)
            b_fake_out = b_disc(b_fake.detach())
            b_d_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2)) / 2

            d_opt.zero_grad()
            d_loss = a_d_loss + b_d_loss
            d_loss.backward()
            d_opt.step()

            # Training Generator
            a_fake_out = a_disc(a_fake)
            b_fake_out = b_disc(b_fake)

            a_g_loss = torch.mean((a_fake_out - 1) ** 2)
            b_g_loss = torch.mean((b_fake_out - 1) ** 2)
            g_gan_loss = a_g_loss + b_g_loss

            a_g_ctnt_loss = (a_real - ba_gen(b_fake)).abs().mean()
            b_g_ctnt_loss = (b_real - ab_gen(a_fake)).abs().mean()
            g_ctnt_loss = a_g_ctnt_loss + b_g_ctnt_loss

            g_opt.zero_grad()
            g_loss = g_gan_loss + g_ctnt_loss
            g_loss.backward()
            g_opt.step()

            if i % 50 == 0:
                plot.append([round(a_d_loss.item(), 4), round(a_g_loss.item(), 4), round(a_g_ctnt_loss.item(), 4), round(b_d_loss.item(), 4), round(b_g_loss.item(), 4), round(b_g_ctnt_loss.item(), 4)])
                print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                      + " it: " + str(i) + "/" + str(iters_per_epoch)
                      + "\ta_d_loss:" + str(round(a_d_loss.item(), 4))
                      + "\ta_g_loss:" + str(round(a_g_loss.item(), 4))
                      + "\ta_g_ctnt_loss:" + str(round(a_g_ctnt_loss.item(), 4))
                      + "\tb_d_loss:" + str(round(b_d_loss.item(), 4))
                      + "\tb_g_loss:" + str(round(b_g_loss.item(), 4))
                      + "\tb_g_ctnt_loss:" + str(round(b_g_ctnt_loss.item(), 4)))

        torch.save(ab_gen.state_dict(), os.path.join(model_path, 'ab_gen.pkl'))
        torch.save(ba_gen.state_dict(), os.path.join(model_path, 'ba_gen.pkl'))
        torch.save(a_disc.state_dict(), os.path.join(model_path, 'a_disc.pkl'))
        torch.save(b_disc.state_dict(), os.path.join(model_path, 'b_disc.pkl'))

        generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen, samples_path, epoch=epoch + 1)

    plot.close()
    generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen, samples_path)




