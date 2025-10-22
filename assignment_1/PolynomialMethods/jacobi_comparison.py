# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jacobi
from typing import Literal


def a(alpha: float, beta: float, n1: Literal[-1] | Literal[0] | Literal[1], n2: int):
    if n1 == -1 and n2 == 0:
        return 0

    if n1 == 0 and n2 == 0:
        return 0

    match n1:
        case -1:
            return (2 * (n2 + alpha) * (n2 + beta)) / (
                (2 * n2 + alpha + beta + 1) * (2 * n2 + alpha + beta)
            )
        case 0:
            return (alpha**2 - beta**2) / (
                (2 * n2 + alpha + beta + 2) * (2 * n2 + alpha + beta)
            )
        case 1:
            return (2 * (n2 + 1) * (n2 + alpha + beta + 1)) / (
                (2 * n2 + alpha + beta) * (2 * n2 + alpha + beta + 1)
            )


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, n: int):
    if n == 0:
        return np.ones_like(xs)

    if n == 1:
        return 0.5 * (alpha - beta + (alpha + beta + 2) * xs)

    return (
        ((a(alpha, beta, 0, 0) + xs) * jacobi_poly(xs, alpha, beta, n - 1)
        - a(alpha, beta, -1, n - 1) * jacobi_poly(xs, alpha, beta, n - 2)
    )) / a(alpha, beta, 1, n)


# %%
# Comparison for Legendre polynomials (alpha=0, beta=0)
x = np.linspace(-1, 1, 100)
n_max = 4

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot our implementation
for n in range(n_max):
    y_custom = jacobi_poly(x, 0, 0, n)
    ax1.plot(x, y_custom, label=f'P{n}', linewidth=2)

ax1.set_title('Our Implementation (Legendre)')
ax1.set_xlabel('x')
ax1.set_ylabel('P_n(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot scipy implementation
for n in range(n_max):
    y_scipy = jacobi(n, 0, 0)(x)
    ax2.plot(x, y_scipy, label=f'P{n}', linewidth=2, linestyle='--')

ax2.set_title('SciPy Implementation (Legendre)')
ax2.set_xlabel('x')
ax2.set_ylabel('P_n(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot difference
for n in range(n_max):
    y_custom = jacobi_poly(x, 0, 0, n)
    y_scipy = jacobi(n, 0, 0)(x)
    difference = np.abs(y_custom - y_scipy)
    ax3.semilogy(x, difference, label=f'|Δ P{n}|')

ax3.set_title('Absolute Difference (Legendre)')
ax3.set_xlabel('x')
ax3.set_ylabel('|Our - SciPy|')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# %%
# Comparison for Chebyshev polynomials of the first kind (alpha=-0.5, beta=-0.5)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot our implementation
for n in range(n_max):
    y_custom = jacobi_poly(x, -0.5, -0.5, n)
    ax1.plot(x, y_custom, label=f'T{n}', linewidth=2)

ax1.set_title('Our Implementation (Chebyshev)')
ax1.set_xlabel('x')
ax1.set_ylabel('T_n(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot scipy implementation
for n in range(n_max):
    y_scipy = jacobi(n, -0.5, -0.5)(x)
    ax2.plot(x, y_scipy, label=f'T{n}', linewidth=2, linestyle='--')

ax2.set_title('SciPy Implementation (Chebyshev)')
ax2.set_xlabel('x')
ax2.set_ylabel('T_n(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot difference
for n in range(n_max):
    y_custom = jacobi_poly(x, -0.5, -0.5, n)
    y_scipy = jacobi(n, -0.5, -0.5)(x)
    difference = np.abs(y_custom - y_scipy)
    ax3.semilogy(x, difference, label=f'|Δ T{n}|')

ax3.set_title('Absolute Difference (Chebyshev)')
ax3.set_xlabel('x')
ax3.set_ylabel('|Our - SciPy|')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# %%
# Numerical comparison - compute max errors
print("Maximum absolute errors:")
print("=" * 40)

print("\nLegendre Polynomials (α=0, β=0):")
for n in range(n_max):
    y_custom = jacobi_poly(x, 0, 0, n)
    y_scipy = jacobi(n, 0, 0)(x)
    max_error = np.max(np.abs(y_custom - y_scipy))
    print(f"  P_{n}: {max_error:.2e}")

print("\nChebyshev Polynomials (α=-0.5, β=-0.5):")
for n in range(n_max):
    y_custom = jacobi_poly(x, -0.5, -0.5, n)
    y_scipy = jacobi(n, -0.5, -0.5)(x)
    max_error = np.max(np.abs(y_custom - y_scipy))
    print(f"  T_{n}: {max_error:.2e}")

# %%
