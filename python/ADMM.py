#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from energy import Energy


class Params(object):
    def __init__(self):
        self._v = False
        self._imageScale = 1
        self._noise = 8

        self._kernelSize = 3

        self._eps = 1e-10
        self._mu2 = 50
        self._mu1Fact = 1.01
        self._mu2Fact = 1.01
        self._solvePCG = True
        self._maxLoops = 1000

    @property
    def _kernelSize(self):
        return self.__kernelSize

    @_kernelSize.setter
    def _kernelSize(self, n):
        self.__kernelSize = n

        self._kernel = np.ones((n,)*2)
        self._kernel[(n//2,)*2] = 0


def compactness_seg_prob_map(img, prob_map, params=None):
    """
    Wrapper function performing the graph cut and the ADMM
    :param img: The gray-scale image to segment
    :param prob_map: the probabilities for segmentation
    :param params: The Params object
    :return: admm_segmentation, graph_cut segmentation, res code
    """
    if params is None:
        params = Params()

    small_eps = 1e-6

    N = img.size
    W = compute_weights(img, params._kernel, params._sigma, params._eps)
    L = sp.sparse.spdiags(W.sum(0), 0, N, N) - W

    # Initial Graph Cut
    priors = prob_map.ravel()

    u_0 = np.zeros((N, 2))
    u_0[:, 0] = -np.log(small_eps + (1 - priors))
    u_0[:, 1] = -np.log(small_eps + priors)

    y_0, E, eg = graph_cut(params, W, u_0, params._kernel, N)
    seg_0 = y_0.reshape(img.shape)

    # ADMM
    p = np.zeros((N, 2))
    p[:, 0] = np.log(small_eps + 1 - prob_map.ravel())
    p[:, 1] = np.log(small_eps + prob_map.ravel())
    y, res = admm(params, y_0, N, L, u_0, p, eg, W)

    seg = y.reshape(img.shape)

    return seg, seg_0, res


def admm(params, y_0, N, L, u_0, p, eg, W):
    _mu1 = params._mu1
    _mu2 = params._mu2
    _lambda = params._lambda

    y, V = y_0.copy(), u_0.copy()
    c, o = np.sum(y), np.ones(N)
    u = np.zeros((N, 2))
    v = 0
    tt = y.T.dot(L.dot(y))  # Careful with the order, since L is sparse. np.dot is unaware of that fact.

    y = np.asarray([1-y, y]).T
    z = np.zeros(y.shape)

    δ = np.ones((2, 2)) - np.diag((1,) * 2)
    Φ = sp.sparse.kron(W, δ)

    cost_1_prev = 0
    for i in range(params._maxLoops):
        # Debug metrics:
        if params._v:
            seg = np.argmax(y, axis=1)
            length = seg.T.dot(L.dot(seg))
            area = seg.sum()
            print("Iteration {:4d}: length = {:5.2f}, area = {:5d}".format(i, length, area))

        # Update z
        alpha = (_lambda / c) * tt

        a = (alpha*L + _mu1 * scipy.sparse.identity(N))
        b = (_mu1 * (y[:, 1] + u[:, 1]) + _mu2 * (c + v))
        if params._solvePCG:
            tmp = sp.sparse.linalg.cg(a, b)[0]
        else:
            tmp = sp.sparse.linalg.spsolve(a, b)

        const = (1 / _mu1) * (1 / _mu2 + N / _mu1) ** -1
        z[:, 1] = tmp - const * np.sum(tmp) * o
        z[:, 0] = 1 - z[:, 1]

        # Update c
        rr = z[:, 1].T.dot(L.dot(z[:, 1]))
        beta = .5 * _lambda * tt * rr

        qq = np.sum(z[:, 1]) - v

        eq = [1, -qq, 0, -beta/_mu2]
        R = np.roots(eq)
        R = R[np.isreal(R)]

        if len(R) == 0:
            print("No roots found...")
            params._lambda /= 10
            return admm(y_0, N, L, u_0, p, eg)

        c = np.real(np.max(R))

        # Update y
        gamma = .5 * (_lambda / c) * rr

        q = z - u
        a = p + params._mu1*(.5 + q)
        a = a + 2 * _lambda * Φ.dot(y.ravel()).reshape(y.shape)

        V[:, 1] = (p[:, 1] + _mu1 * (u[:, 1] - z[:, 1] + .5)).T / (gamma + params._lambda0)
        eg.set_unary(V)
        E = eg.minimize()
        if params._v and i == 0:
            print("E for first iteration: {}".format(E))
        y[:, 1] = eg.get_labeling()
        y[:, 0] = 1 - y[:, 1]

        tt = y[:, 1].T.dot(L.dot(y[:, 1]))

        # Update Lagrangian multipliers
        # u = u + (y[:, 1] - z[:, 1])
        u = u + (y - z)
        v = v + (c - np.sum(z[:, 1]))
        _mu1 *= params._mu1Fact
        _mu2 *= params._mu2Fact

        cost_1 = p[:, 1].T.dot(y[:, 1])
        if cost_1_prev == cost_1:
            if params._v:
                print(i)
            break
        cost_1_prev = cost_1

    return y[:, 1], 0


def graph_cut(params, W, u_0, kernel, N):
    """
    Perform the graph cut for the initial segmentation.
    The current implementation is not fully functional, but the results for RIGHTVENT_MRI
    are usable to develop the rest of the algorithm.
    :param W: The weights matrices computed previously
    :param u_0: The unary weights for the graphcut: based on prob_map
    :param kernel: The kernel used
    :param N: size of the image
    :return: The segmentation as a vector, the Energy
    """
    eg = Energy(N, np.count_nonzero(kernel)*N)
    eg.set_neighbors(W)
    eg.set_unary(u_0)
    E = eg.minimize()
    if params._v:
        print(E)
    y_0 = eg.get_labeling()

    return y_0, E, eg


def compute_weights(img, kernel, sigma, eps):
    """
    This function compute the weights of the graph representing img.
    The weights 0 <= w_i <= 1 will be determined from the difference between the nodes: 1 for identical value,
    0 for completely different.
    :param img: The image, as a (n,n) matrix.
    :param kernel: A binary mask of (k,k) shape.
    :param sigma: Parameter for the weird exponential at the end.
    :param eps: Other parameter for the weird exponential at the end.
    :return: A float valued (n^2,n^2) symmetric matrix. Diagonal is empty
    """
    W, H = img.shape
    N = img.size
    X = img.ravel()

    KW, KH = kernel.shape
    K = int(np.sum(kernel))  # 0 or 1

    A = np.pad(np.arange(N).reshape(img.shape), ((KW//2, KW//2), (KH//2, KH//2)), 'constant', constant_values=-1)
    neighs = np.zeros((K, N), np.int64)

    k = 0
    for i in range(KW):
        for j in range(KH):
            if kernel[i, j] == 0:
                continue

            T = A[i:i+W, j:j+H]
            neighs[k, :] = T.ravel()
            k += 1

    T1 = np.tile(np.arange(N), K)
    T2 = neighs.flatten()
    Z = T1 <= T2  # We need to reverse the test from the Matlab version
    # Matlab delete the Z, we keep them.
    T1 = T1[Z]
    T2 = T2[Z]

    '''
    This represent the difference between the nodes
    1 for the identical values, 0 for completely different ones
    '''
    diff = (1 - eps) * np.exp(-sigma * (X[T1] - X[T2])**2) + eps
    M = sp.sparse.csc_matrix((diff, (T1, T2)), shape=(N, N))

    return M + M.T


if __name__ == "__main__":
    print(params._kernel)
