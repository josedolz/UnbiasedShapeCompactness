#!/usr/bin/env python3

import numpy as np
import scipy as sci
import scipy.sparse
import maxflow

class Energy(object):
    def __init__(self, nodes, connections):
        self._g = maxflow.Graph[float](nodes, connections)
        self._N = nodes
        self._g.add_nodes(self._N)

        self._was_minimized = False

    def set_neighbors(self, W):
        W = scipy.sparse.coo_matrix(W)

        for i,j,w  in zip(W.row, W.col, W.data):
            self._g.add_edge(i, j, w, w)

    def set_unary(self, u_0):
        for i, u in enumerate(u_0):
            self._g.add_tedge(i, u[1], u[0])

    def minimize(self):
        return self._g.maxflow()

    def get_labeling(self):
        labels = self._g.get_grid_segments(np.arange(self._N))

        return labels
