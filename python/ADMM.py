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
    input = (np.arange(16)+1).reshape(4, 4)
    input[1:3, 1:3] = 17

    k = np.ones((3,3))
    k[1, 1] = 0

    output = compute_weights(input, k, 100, 1e-10).toarray()
    print(output)
