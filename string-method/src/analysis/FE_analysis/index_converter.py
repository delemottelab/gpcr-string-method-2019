from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
import utils
logger = logging.getLogger("indexconverter")


class IndexConverter(object):
    def __init__(self, ndim, ngrid):
        self.ndim = ndim
        self.ngrid = ngrid
        self._modulus = [(ngrid - 1) ** (ndim - j - 1) for j in range(ndim)]
        self._zerodim = np.zeros((self.ndim,))
        self.nbins = int(np.rint((ngrid - 1) ** ndim))

    def convert_to_vector(self, grid):
        if grid.shape[0] != self.ngrid - 1:
            raise Exception("Wrong dimension of grid. Expect length fo %s got %s" % (self.ngrid - 1, grid.shape[0]))
        vector = np.empty((self.nbins,))
        for bin_idx in range(self.nbins):
            vector[bin_idx] = grid[tuple(self.convert_to_grid_idx(bin_idx))]
        return vector

    def convert_to_grid(self, vector):
        grid_shape = tuple(np.zeros(self.ndim).astype(int) + (self.ngrid - 1))
        if len(vector.shape) > 1:
            grids = np.empty((len(vector),) + grid_shape)
            for idx, v in enumerate(vector):
                grids[idx] = self.convert_to_grid(v)
            return grids
        else:
            grid = np.zeros(grid_shape)
            for idx in range(len(vector)):
                grid[tuple(self.convert_to_grid_idx(idx))] = vector[idx]
            return grid

    def convert_to_grid_idx(self, bin_idx):
        if bin_idx >= self.nbins or bin_idx < 0:
            print(self.nbins, self.ndim, self.nbins ** self.ndim)
            raise Exception("Invalid index %s. You are probably outside the grid..." % bin_idx)
        grid_idx = ((self._zerodim + bin_idx) / self._modulus) % (self.ngrid - 1)
        return grid_idx.astype(int)

    def convert_to_bin_idx(self, grid_idx):
        bin_idx = utils.rint(np.sum(grid_idx * self._modulus))
        if bin_idx >= self.nbins or bin_idx < 0:
            raise Exception(
                "Invalid bin index %s. You are probably outside the grid. Size:%s" % (bin_idx, self.nbins))
        return bin_idx
