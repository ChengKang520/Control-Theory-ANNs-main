import torch
from torch.optim.optimizer import Optimizer
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

class HPFSGDOptimizer(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.002, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, hpf_sgd=False):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, hpf_sgd=hpf_sgd)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(HPFSGDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HPFSGDOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            hpf_sgd = group['hpf_sgd']

            for p in group['params']:
                # print("##############   Print p   ##############")
                # print(p.name)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if hpf_sgd == True:

                    self.xnt_fiftyHz_bandstop = torch.zeros_like(p.data)
                    self.xm1_fiftyHz_bandstop = torch.zeros_like(p.data)
                    self.xm2_fiftyHz_bandstop = torch.zeros_like(p.data)
                    self.ynt_fiftyHz_bandstop = torch.zeros_like(p.data)
                    self.ym1_fiftyHz_bandstop = torch.zeros_like(p.data)
                    self.ym2_fiftyHz_bandstop = torch.zeros_like(p.data)

                    param_state = self.state[p]
                    if 'HPF_buffer_xnt' not in param_state:
                        HPF_buffer_xnt = param_state['HPF_buffer_xnt'] = torch.zeros_like(p.data)
                        HPF_buffer_xm1 = param_state['HPF_buffer_xm1'] = torch.zeros_like(p.data)
                        HPF_buffer_xm2 = param_state['HPF_buffer_xm2'] = torch.zeros_like(p.data)
                        HPF_buffer_ynt = param_state['HPF_buffer_ynt'] = torch.zeros_like(p.data)
                        HPF_buffer_ym1 = param_state['HPF_buffer_ym1'] = torch.zeros_like(p.data)
                        HPF_buffer_ym2 = param_state['HPF_buffer_ym2'] = torch.zeros_like(p.data)

                    else:

                        HPF_buffer_xm1 = param_state['HPF_buffer_xm1']
                        HPF_buffer_xm2 = param_state['HPF_buffer_xm2']
                        HPF_buffer_ym1 = param_state['HPF_buffer_ym1']
                        HPF_buffer_ym2 = param_state['HPF_buffer_ym2']
                        HPF_buffer_xnt = d_p.clone()

                        HPF_buffer_ynt = 0.49685836 * (-0.0 * HPF_buffer_ym1 + 1.0 * HPF_buffer_ym2 + 1.0 * HPF_buffer_xnt + 0.99371673 * HPF_buffer_xm1 - 0.00628326 * HPF_buffer_xm2)  #  Fs: 500; [Fc1: 125 Fc2: 249]

                        # HPF_buffer_ynt = 0.64452430 * (-0.0 * HPF_buffer_ym1 + 1.0 * HPF_buffer_ym2 + 1.0 * HPF_buffer_xnt + 0.71095140 * HPF_buffer_xm1 - 0.28904860 * HPF_buffer_xm2)  #  Fs: 500; [Fc1: 125 Fc2: 250]

                        # HPF_buffer_ynt = 0.71636932 * (-0.0 * HPF_buffer_ym1 + 1.0 * HPF_buffer_ym2 + 1.0 * HPF_buffer_xnt - 0.38321856 * HPF_buffer_xm1 - 0.43278642 * HPF_buffer_xm2)  #  Fs: 500; [Fc1: 10 Fc2: 200]

                        param_state['HPF_buffer_xm2'] = HPF_buffer_xm1
                        param_state['HPF_buffer_xm1'] = HPF_buffer_xnt
                        param_state['HPF_buffer_ym2'] = HPF_buffer_ym1
                        param_state['HPF_buffer_ym1'] = HPF_buffer_ynt


                    # print("############################")
                    # print(d_p.size())
                    # self.lpf_compute(d_p)

                p.data.add_(-group['lr'], HPF_buffer_ynt)

        return loss
