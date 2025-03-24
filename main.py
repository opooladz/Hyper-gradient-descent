import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.optimizer import Optimizer

# -----------------------------------------------------------------------------
# Custom Hypergradient Descent Optimizer (Simple Version)
# -----------------------------------------------------------------------------
class HyperGradientDescent(Optimizer):
    r"""Hypergradient Descent with Momentum.
    
    The update rule is:
    
        xₜ₊₁ = xₜ - Pₜ * ∇f(xₜ) + βₜ (xₜ - xₜ₋₁)
    
    where the scalars Pₜ and βₜ are updated using AdaGrad–like steps.
    """
    def __init__(self, params, lr=0.001, beta_lr=0.1, adagrad_eps=1e-12,
                 initial_P=0.001, initial_beta=0.95):
        defaults = dict(lr=lr, beta_lr=beta_lr, adagrad_eps=adagrad_eps,
                        initial_P=initial_P, initial_beta=initial_beta)
        super(HyperGradientDescent, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['prev'] = p.data.clone()  # Store xₜ₋₁
                state['P'] = torch.full_like(p.data, initial_P)
                state['G'] = torch.zeros_like(p.data)
                state['beta'] = torch.full_like(p.data, initial_beta)
                state['Gm'] = torch.full_like(p.data, 1e-4)

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
                current_param = p.data.clone()
                P = state['P']
                beta = state['beta']
                
                # Candidate update: xₜ - P * grad + β * (xₜ - xₜ₋₁)
                candidate = current_param - (P * grad) + (beta * (current_param - state['prev']))
                
                # For simplicity, we use the current gradient as surrogate for the candidate gradient.
                candidate_grad = grad  
                denom = grad.pow(2).sum() + 1e-12
                
                # Hypergradient update for P (scalar version)
                gr = - torch.dot(candidate_grad.view(-1), grad.view(-1)) / denom
                state['G'] += gr.item()**2
                new_P = P - lr * gr.item() / (math.sqrt(state['G']) + adagrad_eps)
                state['P'] = new_P
                
                # Hypergradient update for β (scalar version)
                diff = current_param - state['prev']
                gm = torch.dot(candidate_grad.view(-1), diff.view(-1)) / denom
                state['Gm'] += gm.item()**2
                new_beta = beta - beta_lr * gm.item() / (math.sqrt(state['Gm']) + adagrad_eps)
                state['beta'] = max(0.0, min(new_beta, 0.9995))
                
                state['prev'] = current_param
                p.data.copy_(candidate)
        return loss

# -----------------------------------------------------------------------------
# Define the Rosenbrock Function
# -----------------------------------------------------------------------------
def rosenbrock(tensor, lib=torch):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2

# -----------------------------------------------------------------------------
# Simple Rosenbrock Model
# -----------------------------------------------------------------------------
class RosenbrockModel(nn.Module):
    def __init__(self, init_val=None):
        super(RosenbrockModel, self).__init__()
        if init_val is None:
            # A common starting point for Rosenbrock is (-1.2, 1.0)
            init_val = torch.tensor([-1.2, 1.0])
        self.x = nn.Parameter(init_val.clone())

    def forward(self):
        return rosenbrock(self.x, lib=torch)

# -----------------------------------------------------------------------------
# Run the Optimization and Plot the Loss
# -----------------------------------------------------------------------------
num_iters = 200
model = RosenbrockModel()
optimizer = HyperGradientDescent(model.parameters(), lr=0.001, beta_lr=0.1,
                                 initial_P=0.001, initial_beta=0.95)

loss_history = []
for i in range(num_iters):
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# Plot loss over iterations (log-scale for clarity)
plt.figure(figsize=(8, 6))
plt.plot(loss_history, label='Hypergradient Descent')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Rosenbrock Function Optimization', fontsize=16)
plt.yscale('log')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
