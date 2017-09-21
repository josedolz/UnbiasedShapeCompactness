#!/usr/bin/env python3

import graph_tool as gt
import numpy as np
import scipy as sci


def compactness_seg_prob_map(img, prob_map, P):
    """
    Dummy function for the segmentation
    """
    small_eps = 1e-6

    H, W = img.shape
    N = img.size

    X = img.copy()

    M = compute_weights(img, P["kernel"], P["sigma"], P["eps"])

    return prob_map >= 0.5, prob_map >= 0.5, 0


def compute_weights(img, kernel, sigma, eps):
    W, H = img.shape
    N = img.size
    X = img.copy()

    KW, KH = kernel.shape
    K = int(np.sum(kernel))  # 0 or 1

    A = np.pad(np.arange(N).reshape(img.shape), ((KW//2, KW//2), (KH//2, KH//2)), 'constant', constant_values=-1)
    neighs = np.zeros((N, K), np.int64)

    k = 0
    for i in range(KW):
        for j in range(KH):
            if kernel[i, j] == 0:
                continue

            T = A[i:i+W, j:j+H]
            neighs[:, k] = T.flat[:]
            k += 1

    T1 = np.tile(np.arange(N), K)
    T2 = neighs.flat.copy()
    Z = T1 <= T2  # We need to reverse the test from the Matlab version
    # Indeed, we are selecting the true ones, whereas in the Matlab they are deleted
    T1 = T1[Z]
    T2 = T2[Z]

    '''
    This represent the difference between the nodes
    1 for the identical values, 0 for complete different ones
    '''
    diff = (1 - eps) * np.exp(-sigma * (X.flat[:][T1] - X.flat[:][T2])**2) + eps
    assert(np.min(diff) >= 0)
    assert(np.min(diff) <= 1)
    assert(np.count_nonzero(diff) == np.count_nonzero(Z))

    M = sci.sparse.csc_matrix((diff, (T1, T2)), shape=(N,N))

    return M + M.T