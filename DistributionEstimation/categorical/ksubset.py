__author__ = 'wangsw'
# class for k-subset mechanism

import numpy as np
import numpy.random as r
from . import brr
from ... import utils

class KSS:
    name = 'KSS'
    ep = 0.0    # privacy budget epsilon
    ps = None  # points
    ds = None  # distance oracle
    k = 0 # subset size

    dis = 0.0  # distance between two points
    d = 0 # number of points
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false

    def __init__(self, ps, ds, ep, k):
        self.ep = ep
        self.ps = ps
        self.ds = np.copy(ds)
        self.k = k
        self.d = self.ps.shape[0]
        self.dis = np.min(self.ds)
        #print('ksubset', self.ds)
        for i in range(0, self.d):
            self.ds[i, i] = 0.0
        self.__setparams()


    def __setparams(self):
        self.trate = (self.k*np.exp(self.ep*self.dis))/(self.k*np.exp(self.ep*self.dis)+self.d-self.k)
        self.frate = (self.k*np.exp(self.ep*self.dis)*(self.k-1)/(self.d-1)+(self.d-self.k)*self.k/(self.d-1))/(self.k*np.exp(self.ep*self.dis)+self.d-self.k)

    def randomizer(self, secret):
        pub = []
        p = r.random(1)
        l = list(range(0, self.d))
        del(l[secret])
        if p <= self.trate:
            pub.append(secret)
            pub.extend(utils.reservoirsample(l, self.k-1))
        else:
            pub.extend(utils.reservoirsample(l, self.k))
        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        fs = np.array([(hits[i]-n*self.frate)/(self.trate-self.frate) for i in range(0, self.d)])
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical l2-norm error bound
        return (self.trate*(1.0-self.trate)+(self.d-1)*self.frate*(1-self.frate))/(n*(self.trate-self.frate)*(self.trate-self.frate))
