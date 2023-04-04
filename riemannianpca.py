import numpy as np
import pymanopt
import torch
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions
from pyriemann.utils.distance import distance_riemann


def nearest_neighbors(C):
    """
    returns distances and neighbors for each sample
    """
    n_samples = len(C)
    tri = np.tri(n_samples, k=-1)
    line_idx, col_idx = tri.nonzero()
    sample_distances = np.zeros(shape=(n_samples, n_samples))
    for i, j in zip(line_idx, col_idx):
        sample_distances[i, j] = distance_riemann(C[i], C[j])
        sample_distances[j, i] = sample_distances[i, j]

    n_ = np.argsort(sample_distances, axis=1)
    neighbors = n_[:, 1:]
    distances = np.zeros(shape=(n_samples, n_samples - 1))
    for i in range(n_samples):
        distances[i, :] = sample_distances[i, [neighbors[i, :]]]
    return distances, neighbors


def kneighbors(k, neighbors):
    """
    return k adjacency matrix from complete neighborhood graph
    """
    n_samples = neighbors.shape[0]
    nn = np.zeros(shape=(n_samples, n_samples))
    for i in range(n_samples):
        for j in range(k):
            nn[i, neighbors[i, j]] = 1
    return nn


def get_cost_pytorch(manifold, C, y, k=3):
    """Create autograd cost function for a set of covariance matrices C

    Parameters
    ----------
    manifold; Manifold
        Target manifold for optimization
    C: array, shape (2 * n_samples, dim, dim)
        Covariance matrices to decompose in subspaces
    y: array, shape (2 * n_samples)
        Class label (-1 or +1) associated with C
    k: int
        nearest neighbors consider for optimization

    Returns
    -------
    cost: function
        Cost function for pymanopt Problem
    """
    n_samples, _, _ = C.shape
    distances, neighbors = nearest_neighbors(C)
    nn_ = torch.from_numpy(kneighbors(k, neighbors))
    tri = torch.tril(torch.ones(n_samples, n_samples), diagonal=-1)
    line_idx, col_idx = torch.nonzero(tri, as_tuple=True)
    C_, y_ = torch.from_numpy(C), torch.from_numpy(y)

    @pymanopt.function.pytorch(manifold)
    def cost_pytorch(U):
        cost = 0.0
        for i, j in zip(line_idx, col_idx):
            if nn_[i, j] != 0:
                ei, ev = torch.linalg.eigh(
                    torch.matmul(torch.matmul(torch.transpose(U, 1, 0), C_[i]), U),
                    UPLO="U",
                )
                W = torch.matmul(
                    torch.matmul(ev, torch.diag(1.0 / torch.sqrt(ei))),
                    torch.transpose(ev, 1, 0),
                )
                C = torch.matmul(
                    torch.matmul(
                        torch.matmul(torch.matmul(W, torch.transpose(U, 1, 0)), C_[j]),
                        U,
                    ),
                    W,
                )
                ei = torch.linalg.eigvalsh(C, UPLO="U")
                sign = y_[i] * y_[j]
                cost += sign * torch.sum(torch.log(ei) ** 2)
        return cost

    return cost_pytorch


def compute_supervised_rpca(
    C,
    subspace_dim,
    y=None,
    backend="pytorch",
    solver="steepest",
    solv_args=None,
    return_log=False,
    init=None,
    k=3,
):
    """Estimate discrimanative Riemannian PCA from high dim covariance matrices

    Parameters
    ----------
    C: array, , shape (2 * n_samples, dim, dim)
        Covariance matrices to decompose in subspaces
    subspace_dim: int or list
        Dimension of the subspaces
    y: array, shape (2 * n_samples)
        Class label (-1 or +1) associated with C, if None assume that data
        are ordered and balanced between class
    backend: str
        Backend to use for Pymanopt, could be autograd, callable, pytorch,
        or pytorch-gpu
    solver: str
        Solver to use: steepest, trust, conj
    solv_args: dict
        dict of solver arguments
    return_log: bool
        return also the log of optimisation problem
    init: None, str, or list
        Initialization for solver: None, 'Id' or 'SPD' or a list of subspaces
    k: int
        nearest neighbors consider for optimization

    Returns
    -------
    U_opt: array
        Optimal decomposition of subspaces, and their associated weight
    logs: list
        list of dict for each iteration
    """
    n_samples, original_dim, _ = C.shape
    manifold = Stiefel(original_dim, subspace_dim)

    if y is None:
        y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))

    if backend == "pytorch":
        cost = get_cost_pytorch(manifold, C, y, k)
        problem = pymanopt.Problem(manifold=manifold, cost=cost)
    else:
        raise NotImplementedError("no such backend yet")

    if solv_args is None:
        solv_args = {"maxtime": float("inf")}

    if solver == "steepest":
        solver = SteepestDescent(**solv_args, log_verbosity=2)
    elif solver == "trust":
        solver = TrustRegions(**solv_args, log_verbosity=2)
    elif solver == "conj":
        solver = ConjugateGradient(**solv_args, log_verbosity=2)
    res = solver.run(problem, initial_point=init)
    Uopt, logs = res.point, res.log

    if return_log:
        return Uopt, logs
    else:
        return Uopt
