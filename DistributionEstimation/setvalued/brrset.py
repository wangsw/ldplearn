__author__ = 'wangsw'
# class for set-valued data binary randomized response 

import numpy as np
import numpy.random as r


class BRR:
    name = 'BRR'
    ep = 0.0    # privacy budget epsilon

    d = 0 # domain size + maximum subset size
    m = 0 # maximum subset size
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.__setparams()

    def __setparams(self):
        self.trate = np.exp(self.ep/(2*self.m))/(np.exp(self.ep/(2*self.m))+1)
        self.frate = 1.0/(np.exp(self.ep/(2*self.m))+1)
        # print(np.exp(self.ep/2), self.trate, self.frate)

    def randomizer(self, secrets):
        pub = np.zeros(self.d, dtype=int)
        for i in range(0, self.d):
            p = r.random(1)
            if secrets[i] > 0:
                if p < self.trate:
                    pub[i] = 1
                else:
                    pub[i] = 0
            else:
                if p < self.frate:
                    pub[i] = 1
                else:
                    pub[i] = 0
        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        fs = np.array([(hits[i]-n*self.frate)/(self.trate-self.frate) for i in range(0, self.d)])
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return (self.trate*(1.0-self.trate)+(self.d-1)*self.frate*(1-self.frate))/(n*(self.trate-self.frate)*(self.trate-self.frate))
