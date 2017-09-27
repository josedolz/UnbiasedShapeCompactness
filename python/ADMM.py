#!/usr/bin/env python3

import graph_tool.all as gt
import numpy as np
import scipy as sci
import scipy.sparse


def compactness_seg_prob_map(img, prob_map, P):
    """
    Dummy function for the segmentation
    """
    small_eps = 1e-6

    H, W = img.shape
    N = img.size

    X = img.copy()

    W = compute_weights(img, P["kernel"], P["sigma"], P["eps"])
    assert(W.sum(0).all() == W.sum(1).all())
    L = sci.sparse.spdiags(W.sum(0), 0, N, N) - W  # Tiny difference on average compared to the Matlab version
    # I assume this is just artifacts from the float approximations

    priors = prob_map.copy()

    u_0 = np.zeros((N, 2))
    u_0[:, 0] = -np.log(small_eps + (1 - priors.flat[:]))
    u_0[:, 1] = -np.log(small_eps + priors.flat[:])

    p = np.log(prob_map.flat[:])

    v_0 = u_0.copy()
    v = v_0.copy()

    g, source, sink = create_graph(W, u_0)
    res = gt.boykov_kolmogorov_max_flow(g, source, sink, g.edge_properties["cap"])
    part = gt.min_st_cut(g, source, g.edge_properties["cap"], res)

    seg_0 = part.get_array().reshape(img.shape)

    seg = prob_map >= 0.5
    return seg, seg_0, 0


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


def create_graph(W, u_0):
    """
    Create the matrix used for the graph cut.
    Creates edges from W and from u_0.
    The source and sink vertex are found with the argmin and argmax on u_0
    :param W: Sparse matrix for the weights between vertex (pixels)
    :param u_0: Two vertors with the weights between the source/sink and each vertex
    :return: A usable graph for the graphcut
    """
    s = np.argmin(u_0[:, 0])
    t = np.argmax(u_0[:, 1])

    g = gt.Graph()
    coo_W = scipy.sparse.coo_matrix(W)

    cap = g.new_edge_property("double")
    for i,j,v  in zip(coo_W.row, coo_W.col, coo_W.data):
        # print "(%d, %d), %s" % (i,j,v)
        e = g.add_edge(i, j)
        cap[e] = v
        e = g.add_edge(j, i)
        cap[e] = v

    for i,p in enumerate(u_0[:, 0]):
        e = g.add_edge(s, i)
        cap[e] = p
        e = g.add_edge(i, s)
        cap[e] = p
    for i,p in enumerate(u_0[:, 1]):
        e = g.add_edge(i, t)
        cap[e] = p
        e = g.add_edge(t, i)
        cap[e] = p

    g.edge_properties["cap"] = cap

    return g, g.vertex(s), g.vertex(t)


if __name__ == "__main__":
    input = (np.arange(16)+1).reshape(4, 4)
    input[1:3, 1:3] = 17

    k = np.ones((3,3))
    k[1, 1] = 0

    output = compute_weights(input, k, 100, 1e-10).toarray()
    L = sci.sparse.spdiags(output.sum(0), 0, 16, 16) - output
    print(output)
