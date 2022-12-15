"""Optimization module"""
import needle as ndl
import numpy as np
from needle.init import zeros
from needle.ops import power_scalar

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # NOTE: everything we do in the optimizer should not backpropagate gradients
        #       so we need to use .data everywhere.
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            grad_with_weight_decay = grad + self.weight_decay * p.data
            if p not in self.u:
                # self.u[p] = zeros(*p.shape)
                self.u[p] = zeros(*p.shape, device = p.device)
            # self.u[p] = self.momentum *  self.u[p] + (1 - self.momentum) * grad
            self.u[p] = self.momentum *  self.u[p] + (1 - self.momentum) * grad_with_weight_decay
            
            # NOTE: ndl.Tensor() is needed to convert float64 to float32. requires_grad=False is important here because
            #       ndl.Tensor() creates Tensor with requires_grad=True by default!!!
            p.data = p.data - self.lr * ndl.Tensor(self.u[p].data, dtype = p.data.dtype, requires_grad=False)
            # p.data = (1 - self.weight_decay) * p.data - self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # NOTE: this is important. Turns out we want to increment t in the beginning, not in the end
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            grad_with_weight_decay = grad + self.weight_decay * p.data

            if p not in self.m:
                # self.m[p] = zeros(*p.shape)
                self.m[p] = zeros(*p.shape, device = p.device)
            if p not in self.v:
                # self.v[p] = zeros(*p.shape)
                self.v[p] = zeros(*p.shape, device = p.device)
            m_t = self.m[p]
            v_t = self.v[p]
            m_t_1 = self.beta1 * m_t + (1 - self.beta1) * grad_with_weight_decay
            v_t_1 = self.beta2 * v_t + (1 - self.beta2) * (grad_with_weight_decay * grad_with_weight_decay)
            self.m[p] = m_t_1
            self.v[p] = v_t_1
            
            m_t_1_hat = m_t_1 / (1 - self.beta1 ** self.t)
            v_t_1_hat = v_t_1 / (1 - self.beta2 ** self.t)

            p.data = p.data - self.lr * ndl.Tensor(m_t_1_hat.data / (power_scalar(v_t_1_hat.data, 1/2) + self.eps), dtype = p.data.dtype, requires_grad=False)
        ### END YOUR SOLUTION
