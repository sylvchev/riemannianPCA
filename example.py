import numpy as np
from pymanopt.optimizers.line_search import AdaptiveLineSearcher
from pyriemann.datasets import make_matrices

from riemannianpca import compute_subspaces

rs = np.random.RandomState(42)
n, n_dim, r_dim = 10, 5, 2
C = make_matrices(2 * n, n_dim, "spd", rs)
y = np.concatenate((np.zeros(n), np.ones(n)))
solv_args = {
    "line_searcher": AdaptiveLineSearcher(),
    "max_iterations": 30,
    "max_time": float("inf"),
}

Uopt, logs = compute_subspaces(
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
