# %% Imports 
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import save_figure, setup_assignment_plotting, style_axes
from numba import njit

plt.rcParams["text.usetex"] = False

# %% Fourier differentiation matrix D 
# using Algorithm 18 from 'Implementing Spectral Methods for Partial Differential Equations' by David A. Kopriva

@njit
def cot(x):
    return 1 / np.tan(x)

# Using negative Sum trick 
@njit
def fourier_diff_matrix(N):
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = 0
        for j in range(N):
            if i != j:
                D[i, j] = 0.5 * (-1)**(i + j) * cot(np.pi * (i - j) / N)
                D[i, i] -= D[i, j]
    return D

# %% Discrete derivative of v(x)
# exact derivative
# %% Convergence rate

def convergence_rate(Ns): 
    errors = np.zeros(len(Ns))
    for i in range(0,len(Ns)):
        N = Ns[i]
        x = np.linspace(0, 2, N, endpoint=False)
        v = np.exp(np.sin(np.pi * x))
        D = np.pi * fourier_diff_matrix(N)
        D_v = D @ v
        dv_exact = np.pi * np.cos(np.pi * x) * v
        errors[i] = np.linalg.norm(D_v - dv_exact, np.inf)

    return errors


if __name__ == "__main__":
    setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_d")
    Ns = 2 ** np.arange(4, 9)
    errors = convergence_rate(Ns)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(Ns, errors, marker='o', label='Error')
    # Reference line of O(N**2)
    ax.loglog(
        Ns,
        errors[0] * (Ns / Ns[0]) ** -2,
        linestyle='--',
        label=r"$O(N^{-2})$",
    )
    style_axes(
        ax,
        title='Convergence Rate of Fourier Spectral Differentiation',
        xlabel='N',
        ylabel='Infinity Norm of Error',
        legend=True,
        grid={'which': 'both'},
    )
    save_figure("convergence_rate", fig=fig)
