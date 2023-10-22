# class for collision mechanism

import numpy as np
import numpy.random as r
import utils
import time


class Collision:
    name = 'COLLISION'
    ep = 0.0    # privacy budget epsilon

    d = 0 # domain size + maximum subset size
    m = 0 # maximum subset size
    t = 0.0 # output domain size
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false
    normalizer = 0.0 # normalizer for proportional probabilities

    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, m, ep, t=None, clip=False):
        self.ep = ep
        self.d = d
        self.m = m
        self.t = t
        self.clip = clip

        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0

        self.__setparams()

    def __setparams(self):
        if self.t == None:
            self.t = self.bestOutputsize(self.d, self.m, self.ep)
        self.normalizer = self.m*np.exp(self.ep)+(self.t-self.m)
        self.trate = np.exp(self.ep)/self.normalizer
        self.frate = 1.0/self.t
        # print(np.exp(self.ep/2), self.trate, self.frate)

    @staticmethod
    def bestOutputsize(d, m, ep):
        t = int(m*np.exp(ep)+2*m-1)
        return t

    def coreRandomizer(self, xs, seed=None):
        z = 0

        if seed == None:
            seed = r.randint(0, 1000000)*(self.d+self.m)

        # get collisiion set
        collisions = []
        for x in xs:
            r.seed(seed+x)
            v = r.randint(0, self.t)
            collisions.append(v)
        collisions = list(set(collisions))

        #sample
        r.seed(None)
        ur = r.random()
        a = 0
        for i in range(self.t):
            if i in collisions:
                a += np.exp(self.ep)/self.normalizer
                if a > ur:
                    z = i
                    break
            else:
                a += (self.normalizer-len(collisions)*np.exp(self.ep))/((self.t-len(collisions))*self.normalizer)
                if a > ur:
                    z = i
                    break
        #print("COLL Randomizer", collisions, z, seed)
        return (z, seed)


    def randomizer(self, secrets, seed=None):
        xs = utils.bitarrayToList(secrets)

        tstart = time.process_time()
        z,seed = self.coreRandomizer(xs, seed)
        self.clienttime += time.process_time()-tstart

        return self.recorder(z, seed)


    def recorder(self, z, seed):
        # record a value as a hit map
        tstart = time.process_time()
        pub = np.zeros(self.d+self.m, dtype=int)
        for i in range(0, (self.d+self.m)):
            r.seed(seed + i)
            v = r.randint(0, self.t)
            if v == z:
                pub[i] = 1
        self.servertime += time.process_time()-tstart
        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        tstart = time.process_time()
        fs = np.array([(hits[i]-n*self.frate)/(self.trate-self.frate) for i in range(0, self.d+self.m)])
        d = self.d
        fs[0:d//2] = fs[0:d//2]+fs[d//2:d]
        fs[d//2:d] = fs[0:d//2]-2*fs[d//2:d]
        self.servertime += time.process_time()-tstart
        return fs/n

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return (self.m*self.trate*(1.0-self.trate)+self.d*self.frate*(1-self.frate))/(n*(self.trate-self.frate)**2)
