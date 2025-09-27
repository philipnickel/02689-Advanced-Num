# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss

from assignment_1.PolynomialMethods.exercise_k import int_matrix
from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_l")

# %%
N = 100
xs, ws = leggauss(N)
u1 = (xs + 1) ** 0
u2 = np.sin((xs + 1))

M = int_matrix(xs)

value_u1 = u1 @ M @ u1
value_u2 = u2 @ M @ u2

print(value_u1)
print(value_u2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, u1, label=r"$u_1(x) = 1$")
ax.plot(xs, u2, label=r"$u_2(x) = \sin(x+1)$")
style_axes(
    ax,
    title="Test functions on Legendre-Gauss nodes",
    xlabel="x",
    ylabel="value",
    legend=True,
)
save_figure("exercise_l", fig=fig, dpi=200)

# %%
