# %%
import numpy as np
import matplotlib.pyplot as plt
# %%


def generate_h(xj, N):
    h = lambda x: (1 / N) * np.sin((N / 2) * (x - xj)) * (1 / np.tan(0.5 * (x - xj)))

    return h


def generate_dh(xj, N):
    h = generate_h(xj, N)
    dh = (
        lambda x: (0.5) * np.cos((N / 2) * (x - xj)) * (1 / np.tan(0.5 * (x - xj)))
        - np.sin((N / 2) * (x - xj)) / (np.sin(0.5 * (x - xj)) ** 2) / 2 * N
    )

    return dh


def generate_diff_mat(xjs):
    N = len(xjs)
    D = np.zeros((N, N))
    for j, xj in enumerate(xjs):
        dh = generate_dh(xj, N)
        D[:, j] = dh(xjs)
        D[j, j] = 0

    return D


hs = []
dhs = []
NN = 10
xjs = np.linspace(0, 2 * np.pi, NN, endpoint=False)
for xj in xjs:
    h = generate_h(xj, NN)
    dh = generate_dh(xj, NN)
    hs.append(h)
    dhs.append(dh)


xs = np.linspace(0, 2 * np.pi, 100)
for j, (h, dh) in enumerate(zip(hs, dhs)):
    ys = h(xs)
    dys = dh(xs)
    plt.plot(xs, ys, label=f"$h_{j}$")
    # plt.plot(xs, dys, label=f"$dh_{j}$")

plt.legend()
# %%

NN = 30
xjs = np.linspace(0, 2, NN, endpoint=False)
plt.plot(xjs, np.exp(np.sin(np.pi * xjs)))
plt.plot(xjs, generate_diff_mat(np.pi * xjs) @ np.exp(np.sin(np.pi * xjs)))
# %%
