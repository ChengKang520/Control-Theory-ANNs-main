"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import argparse
from fuzzypid import FUZZYPIDOptimizer
from pid import PIDOptimizer
from lpfsgd import LPFSGDOptimizer
from hpfsgd import HPFSGDOptimizer
import os


I_pid = 3
I_pid = float(I_pid)
D_pid = 100
D_pid = float(D_pid)

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


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
    parser.add_argument('--lr_decay_type', type=str, default='linearLR', help='Choose Learning Rate Decay Method: LinearLR, CosineAnnealingLR, ExponentialLR')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')

    args = parser.parse_args()
    print('**************************  args  ******************************')
    print(f"arg is: {args}")
    print('**************************  args  ******************************')

    # Directories for storing model and output samples
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    samples_path = args.samples_path
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    # hardcoding these here
    n_epoch = args.n_epoch  # 20
    batch_size = args.bsz  # 256
    n_T = 400  # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = samples_path
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=n_T)
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)

    ##########  optimiser setup  ##########
    if args.controller_type == 'sgd':
        optim = torch.optim.SGD(ddpm.parameters(), lr=args.learning_rate)
    elif args.controller_type == 'sgdm':
        optim = torch.optim.SGD(ddpm.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.controller_type == 'adam':
        optim = torch.optim.Adam(ddpm.parameters(), lr=args.learning_rate)
    elif args.controller_type == 'radam':
        optim = torch.optim.RAdam(ddpm.parameters(), lr=args.learning_rate)
    elif args.controller_type == 'nadam':
        optim = torch.optim.NAdam(ddpm.parameters(), lr=args.learning_rate)
    elif args.controller_type == 'adamw':
        optim = torch.optim.AdamW(ddpm.parameters(), lr=args.learning_rate)
    elif args.controller_type == 'pid':
        optim = PIDOptimizer(ddpm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9,
                             I_pid=I_pid, D_pid=D_pid)
    elif args.controller_type == 'lpfsgd':
        optim = LPFSGDOptimizer(ddpm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                lpf_sgd=True)
    elif args.controller_type == 'hpfsgd':
        optim = HPFSGDOptimizer(ddpm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                hpf_sgd=True)
    elif args.controller_type == 'fuzzypid':
        optim = FUZZYPIDOptimizer(ddpm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                  momentum=0.9, I_pid=I_pid, D_pid=D_pid)

    ##########  Learning Rate Decay Setup  ##########
    if args.lr_decay_type == 'LinearLR':
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.5, total_iters=4)
    elif args.lr_decay_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0)
    elif args.lr_decay_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
    elif args.lr_decay_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
    elif args.lr_decay_type == 'None':
        scheduler = optim


    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            # optim.step()
            scheduler.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, samples_path + f"ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), model_path + f"ddpm_mnist.pth")
