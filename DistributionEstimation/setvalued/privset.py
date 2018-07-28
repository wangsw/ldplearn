__author__ = 'wangsw'
# class for set-valued data PrivSet mechanism

import numpy as np
import numpy.random as r
from scipy.special import comb
import utils

class PrivSet:
    name = 'PRIVSET'
    ep = 0.0    # privacy budget epsilon

    d = 0 # domain size + maximum subset size
    m = 0 # maximum subset size
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false
    normalizer = 0.0 # normalizer for proportional probabilities

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k

        self.__setparams()

    def __setparams(self):
        if self.k == None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        interCount = comb(self.d + self.m, self.k) - comb(self.d, self.k)
        nonInterCount = comb(self.d, self.k)
        normalizer = nonInterCount + interCount * np.exp(self.ep)
        self.normalizer = normalizer
        self.trate = comb(self.d + self.m - 1, self.k - 1) * np.exp(self.ep) / normalizer
        self.frate = (comb(self.d - 1, self.k - 1) + (interCount * self.k - comb(self.d + self.m - 1, self.k - 1) * self.m) * np.exp(self.ep) / self.d) / normalizer
        # print(np.exp(self.ep/2), self.trate, self.frate)

    @staticmethod
    def bestSubsetSize(d, m, ep):
        errorbounds = np.full(d+m, 0.0)
        infos = [None]*(d+m)
        for k in range(1, d):
            interCount = comb(d+m, k)-comb(d,k)
            nonInterCount = comb(d,k)
            normalizer = nonInterCount + interCount*np.exp(ep)
            trate = comb(d+m-1, k-1)*np.exp(ep)/normalizer
            frate = (comb(d-1,k-1)+(interCount*k-comb(d+m-1,k-1)*m)*np.exp(ep)/d)/normalizer
            errorbounds[k] = (trate*(1.0-trate)+(d+m-1)*frate*(1.0-frate))/((trate-frate)*(trate-frate))
            infos[k] = [trate, frate, errorbounds[k]]
        bestk = np.argmin(errorbounds[1:d])+1
        print("best ", [bestk]+infos[bestk])
        return [bestk]+infos[bestk]

    def randomizer(self, secrets):
        pub = np.zeros(self.d+self.m, dtype=int)
        probs = np.full(self.k+1, 0.0)
        for inter in range(0, self.k+1):
            probs[inter] = comb(self.m, inter)*comb(self.d, self.k-inter)/self.normalizer
        probs = probs*np.exp(self.ep)
        probs[0] = probs[0]/np.exp(self.ep)

        for inter in range(1, self.k+1):
            probs[inter] += probs[inter-1]
        p = r.random(1)[0]
        #print("p", p, probs, secrets.tolist())
        sinter = 0
        while probs[sinter] <= p:
            sinter += 1
        pubset = utils.reservoirsample(utils.bitarrayToList(secrets), sinter)+utils.reservoirsample(utils.bitarrayToList(np.ones(self.d+self.m)-secrets), self.k-sinter)
        #print("pubset", pubset)
        for i in range(0, self.d+self.m):
            if i in pubset:
                pub[i] = 1
        return pub


    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        fs = np.array([(hits[i]-n*self.frate)/(self.trate-self.frate) for i in range(0, self.d+self.m)])
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return (self.trate*(1.0-self.trate)+(self.d+self.m-1)*self.frate*(1-self.frate))/(n*(self.trate-self.frate)*(self.trate-self.frate))
