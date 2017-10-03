#!/usr/bin/env python3

import numpy as np
import scipy as sci
import scipy.sparse
import scipy.sparse.linalg
from energy import Energy


def compactness_seg_prob_map(img, prob_map, P):
    """
    Wrapper function performing the graph cut and the ADMM
    :param img: The gray-scale image to segment
    :param prob_map: the probabilities for segmentation
    :param P: Dictionary containing all the parameters.
    :return: admm_segmentation, graph_cut segmentation, res code
    """
    small_eps = 1e-6

    N = img.size

    W = compute_weights(img, P["kernel"], P["sigma"], P["eps"])
    L = sci.sparse.spdiags(W.sum(0), 0, N, N) - W  # Tiny difference on average compared to the Matlab version
    # This is artifacts from the float approximations

    priors = prob_map.flat

    u_0 = np.zeros((N, 2))
    u_0[:, 0] = -np.log(small_eps + (1 - priors[:]))
    u_0[:, 1] = -np.log(small_eps + priors[:])

    p = np.log(prob_map.flat[:])

    v_0 = u_0.copy()
    V = v_0.copy()

    y_0, E, eg = graph_cut(W, u_0, P["kernel"], N)
    seg_0 = y_0.reshape(img.shape)

    y, res = admm(P, y_0, N, L, V, p, eg)
    seg = y.reshape(img.shape)

    return seg, seg_0, res


def admm(P, y_0, N, L, V, p, eg):
    _mu1 = P["mu1"]
    _mu2 = P["mu2"]
    _lambda = P["lambda"]
    res = 0

    y = y_0.copy()
    c = np.sum(y)
    o = np.ones(N)
    u = np.zeros(N)
    v = 0
    tt = y.T.dot(L.dot(y))  # Careful with the order, since L is sparse. np.dot is unaware of that fact.

    cost_1_prev = 0
    for i in range(P["maxLoops"]):
        # Update z
        alpha = (_lambda / c) * tt

        if P["solvePCG"]:
            tmp = 0
        else:
            a = (alpha*L + _mu1 * scipy.sparse.identity(N))
            b = (_mu1 * (y + u) + _mu2 * (c + v))
            tmp = sci.sparse.linalg.spsolve(a, b)

        const = (1 / _mu1) * (1 / _mu2 + N / _mu1) ** -1
        z = tmp - const * np.sum(tmp) * o

        # Update c
        rr = z.T.dot(L.dot(z))
        beta = .5 * _lambda * tt * rr

        qq = np.sum(z) - v

        eq = [1, -qq, 0, -beta/_mu2]
        R = np.roots(eq)
        R = R[np.isreal(R)]

        if len(R) == 0:
            print("No roots found...")
            P["lambda"] /= 10
            return admm(P, y_0, N, L)

        c = np.real(np.max(R))

        # Update y
        gamma = .5 * (_lambda / c) * rr

        V[:, 1] = (p + _mu1 * (u - z + .5)).T / (gamma + P["lambda0"])
        eg.set_unary(V)
        E = eg.minimize()
        if i == 0:
            print("E for first iteration: {}".format(E))
        y = eg.get_labeling()

        tt = y.T.dot(L.dot(y))

        # Update Lagrangian
        u = u + (y - z)
        v = v + (c - np.sum(z))
        _mu1 *= P["mu1Fact"]
        _mu2 *= P["mu2Fact"]

        cost_1 = p.T.dot(y)
        if cost_1_prev == cost_1:
            res = 0
            print(i)
            break
        cost_1_prev = cost_1

    return y, res


def graph_cut(W, u_0, kernel, N):
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
    X = img.flat.copy()

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
            neighs[k, :] = T.flat[:]
            k += 1

    T1 = np.tile(np.arange(N), K)
    T2 = neighs.flat.copy()
    Z = T1 <= T2  # We need to reverse the test from the Matlab version
    # Matlab delete the Z, we keep them.
    T1 = T1[Z]
    T2 = T2[Z]

    '''
    This represent the difference between the nodes
    1 for the identical values, 0 for complete different ones
    '''
    diff = (1 - eps) * np.exp(-sigma * (X[T1] - X[T2])**2) + eps
    M = sci.sparse.csc_matrix((diff, (T1, T2)), shape=(N, N))

    return M + M.T


if __name__ == "__main__":
    input = (np.arange(16) + 1).reshape(4, 4)
    input[1:3, 1:3] = 17

    k = np.ones((3, 3))
    k[1, 1] = 0

    output = compute_weights(input, k, 100, 1e-10).toarray()
    L = sci.sparse.spdiags(output.sum(0), 0, 16, 16) - output
    print(output)
