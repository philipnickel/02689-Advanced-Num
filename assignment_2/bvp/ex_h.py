# %%
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt

from assignment_1.FourierSpectralMethods.exercise_c import (
    diff_matrix as fourier_diff_matrix,
)
from assignment_1.PolynomialMethods.exercise_k import (
    diff_matrix as polynomial_diff_matrix,
)

# %%


Nt = 100
Nx = Nt
t1 = 0
t2 = 1
x1 = 0
x2 = 2 * np.pi

a = 1

ys, _ = leggauss(Nt)

ts, _ = leggauss(Nt)
ts = 0.5 * (t2 - t1) * (ts + 1) + t1

xs, _ = leggauss(Nx)
xs = 0.5 * (x2 - x1) * (xs + 1) + x1

Ts, Xs = np.meshgrid(ts, xs)

sigma = 1
mu = x2 / 2
F = (
    lambda x: 1
    / (sigma * np.sqrt(2 * np.pi))
    * np.exp(-(0.5 * ((x - mu) / sigma) ** 2))
)
Phi = F(Xs - a * Ts)

Dx = (2 / (x2 - x1)) * polynomial_diff_matrix(ys)
Dt = (2 / (t2 - t1)) * polynomial_diff_matrix(ys)

Lt_block = Dt
Lx_block = (a * np.eye(Dx.shape[0])) @ Dx

Lt = np.kron(Lt_block, np.eye(Nt))
Lx = np.kron(np.eye(Nx), Lx_block)

L = Lx + Lt

b = np.zeros_like(Phi)
b[:, 0] = 1
# b[:, -1] = 1
b[0, :] = 1
b[-1, :] = 1
b = b.flatten(order="F")
indices = np.where(b == 1)[0]
L[indices, :] = 0
L[indices, indices] = 1
b[indices] = Phi.flatten(order="F")[indices]

Phi_hat = np.linalg.solve(L, b).reshape(Phi.shape, order="F")

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# Plot Phi
im1 = axs[0].matshow(Phi)
axs[0].set_title("True")
fig.colorbar(im1, ax=axs[0])

# Plot Phi_hat
im2 = axs[1].matshow(Phi_hat)
axs[1].set_title("Predicted")
fig.colorbar(im2, ax=axs[1])

im3 = axs[2].matshow(Phi - Phi_hat)
axs[2].set_title("Error")
fig.colorbar(im3, ax=axs[2])

plt.tight_layout()
