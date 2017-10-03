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
        self._prev_u = np.zeros((self._N, 2))

    def set_neighbors(self, W):
        W = sci.sparse.triu(W.tocoo())

        for i, j, w in zip(W.row, W.col, W.data):
            self._g.add_edge(i, j, w, w)

    def set_unary(self, U):
        if not self._was_minimized:
            for i, u in enumerate(U):
                self._g.add_tedge(i, u[1], u[0])
        else:
            diff = U - self._prev_u
            for i, u in enumerate(diff):
                self._g.add_tedge(i, u[1], u[0])

        self._prev_u[:] = U

    def minimize(self):
        e = self._g.maxflow(reuse_trees=False)
        self._was_minimized = True

        return e

    def get_labeling(self):
        labels = self._g.get_grid_segments(np.arange(self._N))

        return labels
