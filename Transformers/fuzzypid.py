

import torch
from torch.optim.optimizer import Optimizer
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

class FUZZYPIDOptimizer(Optimizer):
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
                 weight_decay=0, nesterov=False, I_pid=5., D_pid=10.):
        self.createFuzzy()
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, I_pid=I_pid, D_pid=D_pid)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FUZZYPIDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FUZZYPIDOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def createFuzzy(self):
        self.fuzz_pend1 = ctrl.Antecedent(np.arange(-0.02, 0.02, 1), 'join1')
        self.fuzz_motor = ctrl.Consequent(np.arange(-0.02, 0.02, 1), 'motor')
        self.fuzz_pend1.automf(7)
        self.fuzz_motor.automf(7)

        #self.fuzz_pend1['average'].view()
        #self.fuzz_motor['average'].view()


        self.rule1 = ctrl.Rule(self.fuzz_pend1['dismal'], self.fuzz_motor['dismal'])
        self.rule2 = ctrl.Rule(self.fuzz_pend1['poor'], self.fuzz_motor['poor'])
        self.rule3 = ctrl.Rule(self.fuzz_pend1['mediocre'], self.fuzz_motor['mediocre'])
        self.rule4 = ctrl.Rule(self.fuzz_pend1['average'], self.fuzz_motor['average'])
        self.rule5 = ctrl.Rule(self.fuzz_pend1['decent'], self.fuzz_motor['decent'])
        self.rule6 = ctrl.Rule(self.fuzz_pend1['good'], self.fuzz_motor['good'])
        self.rule7 = ctrl.Rule(self.fuzz_pend1['excellent'], self.fuzz_motor['excellent'])

        self.pendulum_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule6, self.rule7])
        self.pendulum_fuzz = ctrl.ControlSystemSimulation(self.pendulum_ctrl)

        #self.rule1.view()


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
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            I_pid = group['I_pid']
            D_pid = group['D_pid']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                self.pendulum_fuzz.input['join1'] = d_p.all().to('cpu')
                self.pendulum_fuzz.compute()
                I_pid = I_pid * (self.pendulum_fuzz.output['motor'] + 1)
                D_pid = D_pid * (self.pendulum_fuzz.output['motor'] + 1)

                I_pid = torch.tensor(I_pid)
                D_pid = torch.tensor(D_pid)

                # I_pid.cuda()
                # D_pid.cuda()

                # print("***********  Fuzzy Computing!  I_d_p_fuzzy  ***********")
                # # print(d_p)
                # print(I)
                # print(D)
                # print("***********  Fuzzy Computing!  I_d_p_fuzzy  ***********")


                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(d_p)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - dampening, d_p)

                    if 'grad_buffer' not in param_state:
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = d_p

                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p - g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p - g_buf)
                        self.state[p]['grad_buffer'] = d_p.clone()

                    d_p = d_p.add_(I_pid, I_buf).add_(D_pid, D_buf)
                p.data.add_(-group['lr'], d_p)


        return loss
