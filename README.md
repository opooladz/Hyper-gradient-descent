```markdown
# HyperGradientDescent Optimizer for Rosenbrock Function Optimization

PyTorch implementation of a hypergradient descent optimizer with momentum, applied to minimize the Rosenbrock function. Includes visualization of the optimization trajectory.

## Features

- 🧠 Custom `HyperGradientDescent` optimizer with adaptive learning rate (`P`) and momentum (`β`)
- 📈 Rosenbrock function implementation as a PyTorch model
- ⚡ 200-iteration optimization loop with loss tracking
- 📉 Matplotlib visualization of loss curve (log-scale)

## Installation

```bash
git clone [your-repo-url]
cd [your-repo-directory]
pip install torch matplotlib numpy
```

## Usage

```python
python main.py
```

This will:
1. Initialize parameters at (-1.2, 1.0)
2. Run optimization for 200 iterations
3. Generate loss curve plot in `loss_history.png`

## Results

Example optimization curve (log-scale loss vs iterations):

![Loss Curve](https://github.com/user-attachments/assets/23ad0ce0-7364-4a41-b2fa-4747ab595774)


Typical characteristics:
- Rapid initial convergence 
- Stable long-term optimization
- Final loss < 1e-5 within 200 iterations

## Custom Optimizer Details

### Update Rule
```math
xₜ₊₁ = xₜ - Pₜ·∇f(xₜ) + βₜ(xₜ - xₜ₋₁)
```
Where both `Pₜ` (learning rate) and `βₜ` (momentum) are adapted using AdaGrad-style hypergradient updates.

### Key Features
- **Dual Adaptation**: Simultaneously adapts learning rate and momentum
- **Stability Controls**: 
  - β clamped to [0, 0.9995]
  - AdaGrad-style gradient normalization
- **Memory Efficient**: Only stores previous parameter state

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.001 | Learning rate for P updates |
| `beta_lr` | 0.1 | Learning rate for β updates |
| `initial_P` | 0.001 | Initial learning rate |
| `initial_beta` | 0.95 | Initial momentum coefficient |
| `adagrad_eps` | 1e-12 | Numerical stability term |

## References

1. [Hyperdescent](https://arxiv.org/abs/1901.09017): Gradient-Based Optimization for Hyperparameters (2019)
2. Rosenbrock Function: Classical optimization benchmark

## License

[MIT License](LICENSE)
```
