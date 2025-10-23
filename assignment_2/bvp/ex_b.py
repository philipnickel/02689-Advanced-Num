# %%

# Idea use:
# r: Spectral Polynomial Collocation Method
# theta: Spectral Fourier Collocation Method

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


def solve_bvp(r1, r2, Nr):
    xs, ws = leggauss(Nr)
    rs = 0.5 * (r2 - r1) * (xs + 1) + r1

    Ntheta = Nr
    thetas = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)

    Rs, Theta = np.meshgrid(rs, thetas)
    Phi = (Rs + (r1**2 / Rs)) * np.cos(Theta)

    Dtheta = fourier_diff_matrix(thetas)
    Dr = (2 / (r2 - r1)) * polynomial_diff_matrix(xs)
    Dtheta2 = Dtheta @ Dtheta

    Lr_block = Dr @ Dr + np.diag(1 / rs) @ Dr
    Ltheta_block = Dtheta2

    Lr = np.kron(Lr_block, np.eye(Ntheta))
    Ltheta = np.kron(np.diag(1 / rs**2), Ltheta_block)

    L = Ltheta + Lr

    b = np.zeros_like(Phi)
    b[:, 0] = 1
    b[:, -1] = 1
    b = b.flatten(order="F")
    indices = np.where(b == 1)[0]
    L[indices, :] = 0
    L[indices, indices] = 1
    b[indices] = Phi.flatten(order="F")[indices]

    Phi_hat = np.linalg.solve(L, b).reshape(Phi.shape, order="F")

    return Phi, Phi_hat, Rs, Theta


# %%
Nrs = np.arange(10, 50, step=2)
errors = np.zeros(Nrs.shape[0])

for i, Nr in enumerate(Nrs):
    r1 = 1
    r2 = 10
    Phi, Phi_hat, Rs, Theta = solve_bvp(r1, r2, Nr)
    errors[i] = np.max(np.abs(Phi - Phi_hat))

plt.loglog(Nrs, errors)
# %%
r1 = 1
r2 = 3
Phi, Phi_hat, Rs, Theta = solve_bvp(r1, r2, 20)
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"}, layout="constrained")
con = ax.contourf(Theta, Rs, Phi_hat, 100)
cbar = fig.colorbar(con, ax=ax)
cbar.set_label("Phi")
ax.set_ylim(0, r2)

# %%
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"}, layout="constrained")
con = ax.contourf(Theta, Rs, Phi_hat - Phi, 100)
cbar = fig.colorbar(con, ax=ax)
cbar.set_label("Phi")
ax.set_ylim(0, r2)

# %%
