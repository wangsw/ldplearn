__author__ = 'wangsw'
# class for binary randomized response/RAPPOR

import numpy as np
import numpy.random as r


class BRR:
    name = 'BRR'
    ep = 1.0    # privacy budget epsilon
    ps = None  # points
    ds = None  # distance oracle

    dis = 0.0  # distance between two points
    d = 0 # number of points
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false

    def __init__(self, ps, ds, ep, k=None):
        self.ep = ep
        self.ps = ps
        self.ds = np.copy(ds)
        self.d = self.ps.shape[0]
        self.dis = np.min(self.ds)
        #print('brr', self.ds)
        for i in range(0, self.d):
            self.ds[i, i] = 0.0
        self.__setparams()

    def __setparams(self):
        self.trate = np.exp(self.ep*self.dis/2)/(np.exp(self.ep*self.dis/2)+1)
        self.frate = 1.0/(np.exp(self.ep*self.dis/2)+1)
        # print(np.exp(self.ep*self.dis/2), self.trate, self.frate)

    def randomizer(self, secret):
        pub = []
        for i in range(0, self.d):
            p = r.random(1)
            if i == secret:
                if p < self.trate:
                    pub.append(i)
            else:
                if p < self.frate:
                    pub.append(i)
        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        fs = np.array([(hits[i]-n*self.frate)/(self.trate-self.frate) for i in range(0, self.d)])
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical l2-norm error bound
        return (self.trate*(1.0-self.trate)+(self.d-1)*self.frate*(1-self.frate))/(n*(self.trate-self.frate)*(self.trate-self.frate))
