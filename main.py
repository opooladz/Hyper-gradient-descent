import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.optimizer import Optimizer

# -----------------------------------------------------------------------------
# Custom Hypergradient Descent Optimizer (Corrected Version)
# -----------------------------------------------------------------------------
class HyperGradientDescent(Optimizer):
    r"""Hypergradient Descent with Momentum (Scalar Hyperparameters)."""
    def __init__(self, params, lr=0.001, beta_lr=0.1, adagrad_eps=1e-12,
                 initial_P=0.001, initial_beta=0.95):
        defaults = dict(lr=lr, beta_lr=beta_lr, adagrad_eps=adagrad_eps,
                        initial_P=initial_P, initial_beta=initial_beta)
        super(HyperGradientDescent, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['prev'] = p.data.clone()
                state['P'] = group['initial_P']  # Scalar
                state['G'] = 0.0                 # Scalar accumulator for P
                state['beta'] = group['initial_beta']  # Scalar
                state['Gm'] = 0.0                # Scalar accumulator for beta

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_lr = group['beta_lr']
            adagrad_eps = group['adagrad_eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                current_param = p.data
                P = state['P']
                beta = state['beta']
                
                # Candidate update: xₜ - P * grad + β * (xₜ - xₜ₋₁)
                candidate = current_param - (P * grad) + (beta * (current_param - state['prev']))
                
                # Compute denominator (norm squared of gradients)
                denom = grad.pow(2).sum().item() + 1e-12
                
                # Hypergradient for P
                candidate_grad = grad  # Simplified assumption
                gr = - torch.dot(candidate_grad.view(-1), grad.view(-1)).item() / denom
                state['G'] += gr ** 2
                new_P = P - (lr * gr) / (math.sqrt(state['G']) + adagrad_eps)
                state['P'] = new_P
                
                # Hypergradient for beta
                diff = current_param - state['prev']
                gm = torch.dot(candidate_grad.view(-1), diff.view(-1)).item() / denom
                state['Gm'] += gm ** 2
                new_beta = beta - (beta_lr * gm) / (math.sqrt(state['Gm']) + adagrad_eps)
                new_beta = max(0.0, min(new_beta, 0.9995))
                state['beta'] = new_beta
                
                # Update parameter and previous value
                state['prev'] = current_param.clone()
                p.data.copy_(candidate)
        return loss

# -----------------------------------------------------------------------------
# Define the Rosenbrock Function and Model
# -----------------------------------------------------------------------------
def rosenbrock(tensor, lib=torch):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2

class RosenbrockModel(nn.Module):
    def __init__(self, init_val=None):
        super(RosenbrockModel, self).__init__()
        if init_val is None:
            init_val = torch.tensor([-1.2, 1.0])
        self.x = nn.Parameter(init_val.clone())
    def forward(self):
        return rosenbrock(self.x)

# -----------------------------------------------------------------------------
# Optimization and Plotting
# -----------------------------------------------------------------------------
num_iters = 2000
model = RosenbrockModel()
optimizer = HyperGradientDescent(model.parameters(), lr=0.00001, beta_lr=0.1,
                                 initial_P=0.001, initial_beta=0.95)

loss_history = []
for i in range(num_iters):
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

plt.figure(figsize=(8, 6))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Rosenbrock Function Optimization')
plt.yscale('log')
plt.grid(True)
plt.show()
