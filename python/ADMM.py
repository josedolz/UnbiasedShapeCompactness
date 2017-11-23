#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from energy import Energy

import matplotlib.pyplot as plt


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
        self._GC = False
        self._maxLoops = 1000

        self._crf_loops = 1000
        self._crf_tol = 1e-1
        self._delta_B = 250
        self._e = .1
        self._init_graph_cut = False
        self._binarize_update = False

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

    unary_0 = np.zeros((2, N))
    unary_0[0, :] = -np.log(ε + (1 - priors))
    unary_0[1, :] = -np.log(ε + priors)

    y_0, E, eg = graph_cut(params, W, unary_0.T, params._kernel, N)
    seg_gc = y_0.reshape(img.shape)

    # ADMM
    u = np.zeros((2, N))
    u[0, :] = np.log(ε + 1 - prob_map.ravel())
    u[1, :] = np.log(ε + prob_map.ravel())
    if params._init_graph_cut:
        y, metrics = admm(params, y_0, N, L, unary_0, u, W, eg, img)
    else:
        y, metrics = admm(params, priors, N, L, unary_0, u, W, eg, img)

    final_seg = y.reshape(img.shape)

    return final_seg, seg_gc, metrics


# Careful with the order, since L is sparse. np.dot is unaware of that fact.
c_length = lambda label, L: label[1, :].T.dot(L.dot(label[1, :]))


def b_length(label, L):
    seg = np.argmax(label, axis=0)
    return seg.T.dot(L.dot(seg))


def binarize(y, e):
    r = np.zeros(y.shape)
    r[y <  .5] = e
    r[y >= .5] = 1 - e
    return r


def admm(params, y_0, N, L, unary_0, u, W, eg, img):
    if params._GC:
        y = y_0 >= .5
    else:
        y = binarize(y_0, params._e)

    λ, μ1, μ2 = params._lambda, params._mu1, params._mu2

    y = np.asarray([1-y, y])
    unary = unary_0.copy()
    z = np.zeros((2, N))

    s = np.sum(y[1, :])
    ν1, ν2 = np.zeros((2, N)), 0
    tt = b_length(y, L)

    δ = np.ones((2, 2)) - np.diag((1,) * 2)
    Φ = sp.sparse.kron(W, δ)
    # B = np.max(sp.sparse.linalg.eigsh(Φ)[0], 0) + params._delta_B
    B = params._delta_B  # Save quite a lot of time
    print("β for CRF: {:5.2f}".format(B))

    metrics = {'length': [], 'area': [], 'compactness':[], 'crf': [],
                'diff min': [], 'diff max': [], 'diff avg': [], 'diff std': []}

    not_much = 0
    for i in range(params._maxLoops):
        if params._v and i < 10 and False:
            seg = np.argmax(y, axis=1)
            plt.imshow(seg.reshape(img.shape))
            plt.show()

        metrics = update_metrics(params, y, L, metrics, i)

        z = update_z(params, λ, μ1, μ2, N, L, y, ν1, ν2, s, tt)
        rr = c_length(z, L)

        s = update_s(params, λ, μ2, ν2, tt, rr, z)

        # Update y
        q = z - ν1
        F = u + μ1 * (.5 - q)
        γ = .5 * (λ / s) * rr
        # Λ = (γ + params._lambda0)
        Λ = γ
        if params._GC:
            y1 = gc_update(params, unary, F, Λ, eg)
        else:
            y1, metrics = crf_update(params, F, Λ, Φ, B, y, metrics)
        diff = np.abs(y - y1)
        if diff.mean() < 0.01:
            not_much += 1
        print("\tDiff: {:5.6f}".format(diff.mean()))
        metrics['diff min'].append(diff.min())
        metrics['diff max'].append(diff.max())
        metrics['diff avg'].append(diff.mean())
        metrics['diff std'].append(diff.std())

        # Converged ?
        if np.allclose(y, y1, atol=1e-3):
        # if diff.mean() <= 1e-4:
            if params._v:
                print(i)
            break
        y = y1

        tt = b_length(y, L)
        # Update Lagrangian multipliers
        ν1 = ν1 + (y - z)
        ν2 = ν2 + (s - np.sum(z[1, :]))
        μ1 *= params._mu1Fact
        μ2 *= params._mu2Fact

    if not params._GC:
        print("Iterations without much diff: {}".format(not_much))

    return np.argmax(y, axis=0), metrics


def update_metrics(params, y, L, metrics, i):
    l = b_length(y, L)
    seg = np.argmax(y, axis=0)
    area = seg.sum()
    compactness = l**2 / area

    metrics["length"].append(l)
    metrics["area"].append(area)
    metrics["compactness"].append(compactness)

    if params._v and True:
        print("Iteration {:4d}: length = {:5.2f}, area = {:5d}, compactness = {:5.2f}"
                .format(i, l, area, compactness))

    return metrics


def update_z(params, λ, μ1, μ2, N, L, y, ν1, ν2, s, tt):
    α = (λ / s) * tt

    a = α*L + μ1 * sp.sparse.identity(N)
    b = μ1 * (y[1, :] + ν1[1, :]) + μ2 * (s + ν2)
    tmp = sp.sparse.linalg.cg(a, b)[0]

    const = (1 / μ1) * (1 / μ2 + N / μ1) ** -1
    z = np.zeros((2, N))
    z[1, :] = tmp - const * np.sum(tmp) * np.ones(N)
    z[0, :] = 1 - z[1, :]

    return z


def update_s(params, λ, μ2, ν2, tt, rr, z):
    β = .5 * λ * tt * rr

    qq = np.sum(z[1, :]) - ν2

    eq = [1, -qq, 0, -β/μ2]
    R = np.roots(eq)
    R = R[np.isreal(R)]

    if len(R) == 0:
        raise ValueError("No roots found. Try decreasing λ")

    s = np.real(np.max(R))

    return s


def gc_update(params, unary, F, Λ, eg):
    unary[1, :] = F[1, :] / Λ
    eg.set_unary(unary.T)
    _ = eg.minimize()

    y = np.zeros(unary.shape)
    y[1, :] = eg.get_labeling()
    y[0, :] = 1 - y[1, :]

    return y


def crf_update(params, F, Λ, Φ, B, y, metrics):
    for j in range(params._crf_loops):
        a = F + Λ * Φ.dot(y.T.ravel()).reshape(y.T.shape).T

        exp = np.exp(-a/B)
        y2 = y * exp
        y2 = y2 / np.tile(y2.sum(0), 2).reshape(y2.shape)
        assert(np.allclose(y2.sum(0), 1))
        assert(0 <= y2.min() and y2.max() <= 1)

        if np.allclose(y2, y, atol=params._crf_tol):
            y = y2
            break

        y = y2
    metrics["crf"].append(j)
    if params._v and False:
        print("Crf completed in {:3d} iterations".format(j))

    if params._binarize_update:
        y = binarize(y, params._e)

    return y, metrics


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
    # diff = np.ones(len(T1))
    M = sp.sparse.csc_matrix((diff, (T1, T2)), shape=(N, N))

    return M + M.T


if __name__ == "__main__":
    print(params._kernel)
