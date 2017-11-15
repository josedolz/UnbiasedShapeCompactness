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

    ε = 1e-6

    N = img.size
    W = compute_weights(img, params._kernel, params._sigma, params._eps)
    L = sp.sparse.spdiags(W.sum(0), 0, N, N) - W

    # Initial Graph Cut
    priors = prob_map.ravel()

    unary_0 = np.zeros((N, 2))
    unary_0[:, 0] = -np.log(ε + (1 - priors))
    unary_0[:, 1] = -np.log(ε + priors)

    y_0, E, _ = graph_cut(params, W, unary_0, params._kernel, N)
    seg_0 = y_0.reshape(img.shape)

    # ADMM
    u = np.zeros((N, 2))
    u[:, 0] = np.log(ε + 1 - prob_map.ravel())
    u[:, 1] = np.log(ε + prob_map.ravel())
    y, res = admm(params, priors, N, L, u, W)

    seg = y.reshape(img.shape)

    return seg, seg_0, res


def admm(params, y_0, N, L, u, W):
    μ1 = params._mu1
    μ2 = params._mu2
    λ = params._lambda

    y = y_0.copy()
    s = np.sum(y)
    ν1, ν2 = np.zeros((N, 2)), 0
    tt = y.T.dot(L.dot(y))  # Careful with the order, since L is sparse. np.dot is unaware of that fact.

    y = np.asarray([1-y, y]).T
    z = np.zeros(y.shape)

    δ = np.ones((2, 2)) - np.diag((1,) * 2)
    Φ = sp.sparse.kron(W, δ)
    Β = np.max(sp.sparse.linalg.eigsh(Φ)[0], 0)

    cost_1_prev = 0
    for i in range(params._maxLoops):
        # Debug metrics:
        if params._v:
            seg = np.argmax(y, axis=1)
            length = seg.T.dot(L.dot(seg))
            area = seg.sum()
            print("Iteration {:4d}: length = {:5.2f}, area = {:5d}".format(i, length, area))

        # Update z
        α = (λ / s) * tt

        a = (α*L + μ1 * scipy.sparse.identity(N))
        b = (μ1 * (y[:, 1] + ν1[:, 1]) + μ2 * (s + ν2))
        if params._solvePCG:
            tmp = sp.sparse.linalg.cg(a, b)[0]
        else:
            tmp = sp.sparse.linalg.spsolve(a, b)

        const = (1 / μ1) * (1 / μ2 + N / μ1) ** -1
        z[:, 1] = tmp - const * np.sum(tmp) * np.ones(N)
        z[:, 0] = 1 - z[:, 1]

        # Update c
        # rr = z[:,1].T.dot(L.dot(z[:, 1]))
        Z = np.argmax(z, axis=1)
        rr = Z.T.dot(L.dot(Z))
        β = .5 * λ * tt * rr

        qq = np.sum(z[:, 1]) - ν2

        eq = [1, -qq, 0, -β/μ2]
        R = np.roots(eq)
        R = R[np.isreal(R)]

        if len(R) == 0:
            print("No roots found...")
            params._lambda /= 10
            return admm(y_0, N, L, u, W)

        s = np.real(np.max(R))

        # Update y
        γ = .5 * (λ / s) * rr

        q = z - ν1
        a = u + params._mu1 * (.5 - q)
        a = a + 2 * (γ + .5) * Φ.dot(y.ravel()).reshape(y.shape)

        y = y * np.exp(-a/Β)
        y = y / np.repeat(y.sum(1), 2).reshape(y.shape)
        assert(np.allclose(y.sum(1), 1))
        assert(0 <= y.min() and y.max() <= 1)

        tt = y[:, 1].T.dot(L.dot(y[:, 1]))

        # Update Lagrangian multipliers
        ν1 = ν1 + (y - z)
        ν2 = ν2 + (s - np.sum(z[:, 1]))
        μ1 *= params._mu1Fact
        μ2 *= params._mu2Fact

        cost_1 = u[:, 1].T.dot(y[:, 1])
        if cost_1_prev == cost_1:
            if params._v:
                print(i)
            break
        cost_1_prev = cost_1

    return y[:, 1] >= .5, 0


def graph_cut(params, W, unary_0, kernel, N):
    """
    Perform the initial graph cut for the initial segmentation.
    The current implementation is not fully functional, but the results for RIGHTVENT_MRI
    are usable to develop the rest of the algorithm.
    :param W: The weights matrices computed previously
    :param unary_0: The unary weights for the graphcut: based on prob_map
    :param kernel: The kernel used
    :param N: size of the image
    :return: The segmentation as a vector, the Energy
    """
    eg = Energy(N, np.count_nonzero(kernel)*N)
    eg.set_neighbors(W)
    eg.set_unary(unary_0)
    E = eg.minimize()
    if params._v:
        print(E)
    y_0 = eg.get_labeling()

    return y_0, E, eg


def compute_weights(img, kernel, σ, ε):
    """
    This function compute the weights of the graph representing img.
    The weights 0 <= w_i <= 1 will be determined from the difference between the nodes: 1 for identical value,
    0 for completely different.
    :param img: The image, as a (n,n) matrix.
    :param kernel: A binary mask of (k,k) shape.
    :param σ: Parameter for the weird exponential at the end.
    :param ε: Other parameter for the weird exponential at the end.
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
    diff = (1 - ε) * np.exp(-σ * (X[T1] - X[T2]) ** 2) + ε
    M = sp.sparse.csc_matrix((diff, (T1, T2)), shape=(N, N))

    return M + M.T


if __name__ == "__main__":
    print(params._kernel)
