#!/usr/bin/env python3

import graph_tool.all as gt
import numpy as np
import scipy as sci
import scipy.sparse
import scipy.sparse.linalg
import energy as eg


def compactness_seg_prob_map(img, prob_map, P):
    """
    Dummy function for the segmentation
    """
    small_eps = 1e-6

    H, W = img.shape
    N = img.size

    X = img.copy()

    W = commu1te_weights(img, P["kernel"], P["sigma"], P["eps"])
    assert(W.sum(0).all() == W.sum(1).all())
    L = sci.sparse.spdiags(W.sum(0), 0, N, N) - W  # Tiny difference on average compared to the Matlab version
    # I assume this is just artifacts from the float approximations

    priors = prob_map.flat

    u_0 = np.zeros((N, 2))
    u_0[:, 0] = -np.log(small_eps + (1 - priors[:]))
    u_0[:, 1] = -np.log(small_eps + priors[:])

    p = np.log(prob_map.flat[:])

    v_0 = u_0.copy()
    v = v_0.copy()

    y_0, E = graph_cut(W, u_0, P["kernel"], N)
    seg_0 = y_0.reshape(img.shape)

    y = admm(P, y_0, N, L)
    seg = y.reshape(img.shape)

    return seg, seg_0, 0


def admm(P, y_0, N, L):
    mu1 = P["mu1"]
    mu2 = P["mu2"]
    _lambda = P["lambda"]

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
            a = (alpha*L + mu1 * scipy.sparse.identity(N))
            b = (mu1 * (y + u) + mu2 * (c + v))
            print(a.shape)
            print(b.shape)
            tmp = sci.sparse.linalg.spsolve(a, b)

            # assert(np.allclose(np.dot(a, tmp), b))

        const = (1 / mu1) * (1 / mu2 + N / mu1) ** -1
        z = tmp - const * np.sum(tmp) * o

        # Update c
        rr = z.T.dot(L.dot(z))
        beta = .5 * _lambda * tt * rr

        qq = np.sum(z) - v

        eq = [1, -qq, -beta/mu2]
        print(eq)
        R = np.roots(eq)
        R = R[np.isreal(R)]

        break

        if not R:
            print("No roots found...")
            P["lambda"] /= 10
            return admm(P, y_0, N, L)

        break

    return y_0


def graph_cut(W, u_0, kernel, N):
    """
    Perform the graph cut for the initial segmentation.
    The current implementation is not fully functionnal, but the results for RIGHTVENT_MRI
    are usable to develop the rest of the algorithm.
    :param W: The weights matrices commu1ted previously
    :param u_0: The unary weights for the graphcut: based on prob_map
    :param kernel: The kernel used
    :param N: size of the image
    :return: The segmentation as a vector, the Energy
    """
    g = eg.create(N, np.count_nonzero(kernel))
    eg.set_neighbors(g, W)
    eg.set_unary(g, u_0)
    E = eg.minimize(g)
    print(E)
    y_0 = eg.get_labeling(g, N)

    return y_0, E


def commu1te_weights(img, kernel, sigma, eps):
    """
    This function commu1te the weights of the graph representing img.
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
    inmu1t = (np.arange(16)+1).reshape(4, 4)
    inmu1t[1:3, 1:3] = 17

    k = np.ones((3,3))
    k[1, 1] = 0

    outmu1t = commu1te_weights(inmu1t, k, 100, 1e-10).toarray()
    L = sci.sparse.spdiags(outmu1t.sum(0), 0, 16, 16) - outmu1t
    print(outmu1t)
