import matplotlib.pyplot as plt
import numpy as np
from pymanopt.optimizers.line_search import AdaptiveLineSearcher
from pyriemann.datasets import make_matrices

from riemannianpca import compute_supervised_rpca

rs = np.random.RandomState(42)
n, n_dim, r_dim = 10, 10, 4
C = make_matrices(2 * n, n_dim, "spd", rs)
y = np.concatenate((np.zeros(n), np.ones(n)))
solv_args = {
    "line_searcher": AdaptiveLineSearcher(),
    "max_iterations": 30,
    "max_time": float("inf"),
}


U, logs = compute_supervised_rpca(
    C=C,
    subspace_dim=r_dim,
    y=y,
    backend="pytorch",
    solver="steepest",
    solv_args=solv_args,
    return_log=True,
    init=None,
    k=3,
)

# Reduced matrices are :
# U.T @ C[i] @ U

# Plot
plt.figure(figsize=(7, 7))
for i in range(5):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(C[i], cmap=plt.get_cmap("RdBu_r"))
    plt.xticks([])
    if i == 0:
        plt.yticks(np.arange(n_dim))
        ax.tick_params(axis="both", which="major", labelsize=7)
    else:
        plt.yticks([])
    if i == 2:
        plt.title("Cov original space")

for i, pos in enumerate(range(5, 10)):
    ax = plt.subplot(2, 5, pos + 1)
    plt.imshow(U.T @ C[i] @ U, cmap=plt.get_cmap("RdBu_r"))
    plt.xticks([])
    if i == 0:
        plt.yticks(np.arange(r_dim))
        ax.tick_params(axis="both", which="major", labelsize=7)
    else:
        plt.yticks([])
    if i == 2:
        plt.title("Cov reduced space")
plt.show()
