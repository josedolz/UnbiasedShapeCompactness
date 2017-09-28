#!/usr/bin/env python3

import numpy as np
import scipy as sci
import scipy.sparse
import maxflow


def create(nodes, connections):
    g = maxflow.Graph[float](nodes, connections)
    g.add_nodes(nodes)

    return g

def set_neighbors(g, W):
    W = scipy.sparse.coo_matrix(W)

    for i,j,w  in zip(W.row, W.col, W.data):
        g.add_edge(i, j, w, w)

def set_unary(g, u_0):
    for i, u in enumerate(u_0):
        g.add_tedge(i, u[1], u[0])
        # g.add_tedge(i, u[1], u[0])

def minimize(g):
    return g.maxflow()

def get_labeling(g, N):
    labels = g.get_grid_segments(np.arange(N))
    # return np.logical_not(labels)
    return labels