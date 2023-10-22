# class for collision mechanism

import numpy as np
import numpy.random as r
import utils
import time


class Coco:
    name = 'COCO'
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

    def __init__(self, d, m, ep, t=None, debias=False, clip=False):
        self.ep = ep
        self.d = d
        self.m = m
        self.t = t
        self.debias = debias
        self.clip = clip
        self.b = 1 #+0.1*ep
        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0

        self.__setparams()

    def __setparams(self):

        if self.t == None:
            self.t = self.bestOutputsize(self.d, self.m, self.ep)
        m = self.m
        ep = self.ep

        self.normalizer = m*(np.exp(ep)+self.b)+(self.t-2*m)
        self.trate = np.exp(ep)/self.normalizer
        self.frate = self.b/self.normalizer
        self.conflicts = 0
        self.conflict = 0.0
        for i in range(self.m):
            self.conflict += 1-np.power(1-1/(self.t//2), i)
        self.conflict = self.conflict/m
        if not self.debias:
            self.conflict = 0.0
        # print(np.exp(self.ep/2), self.trate, self.frate)
        print("coco", self.t, self.b, self.conflict)

    @staticmethod
    def bestOutputsize(d, m, ep, alpha=-1):
        t = int(m*np.exp(1.0*ep)+2*m-1)
        if alpha >= 0:
            errors = np.zeros(int(2*m*np.exp(1.0*ep)+8*m+1), dtype=float)
            # iterative compute weighted error bounds
            for i in range(2*m+2, len(errors)):
                a = np.exp(ep)
                b = 1
                omega = (a+b)*m+(i-2*m)
                ppr = a/omega
                npr = b/omega
                tpr = ppr+npr
                fpr = 1/i

                cp = np.sum([1.0-np.power(1-2/i, j) for j in range(m)])/m
                mppr = ppr*(1-cp)+0.5*(ppr+npr)*cp
                mnpr = npr*(1-cp)+0.5*(ppr+npr)*cp

                minusVar = m*(mppr+mnpr-(mppr-mnpr)**2)+(d-m)*2*fpr
                minusVar /= np.power(mppr-mnpr, 2)

                plusVar = m*tpr*(1-tpr)+(d-m)*2*fpr*(1-2*fpr)
                plusVar /= np.power(tpr-2*fpr, 2)

                errors[i] = minusVar + alpha*plusVar
            t = 2*m+2+np.argmin(errors[2*m+2:len(errors)])
        if t%2 == 1:
            t += 1
        t = max(t, 2*m+2)
        return t

    def coreRandomizer(self, xs, seed=None):
        if seed == None:
            seed = r.randint(0, 1000000*(self.d+self.m))

        # get collisiion set
                # get collisiion set
        occupied = []
        collisions = []
        r.shuffle(xs)
        for x in xs:
            v = None
            if x < self.d//2:
                # +1
                r.seed(seed+x)
                v = r.randint(0, self.t)
                if v not in occupied:
                    collisions.append(v)
                    occupied.append(v)
                    occupied.append(v-self.t//2 if v >= self.t//2 else v+self.t//2)
                else:
                    self.conflicts += 1
            elif x < self.d:
                # -1
                r.seed(seed+x-self.d//2)
                v = r.randint(0, self.t)
                if v not in occupied:
                    collisions.append(v-self.t//2 if v >= self.t//2 else v+self.t//2)
                    occupied.append(v)
                    occupied.append(v-self.t//2 if v >= self.t//2 else v+self.t//2)
                else:
                    self.conflicts += 1
            else:
                # padded items
                print("xs", xs)
                assert False
        #collisions_pos = list(set(collisions_pos))
        #collisions_neg = list(set(collisions_neg))
        collisionsa = list(set(collisions))
        if len(collisionsa) != len(collisions):
            print("\n\n Wrong !!!\n\n")
        #collisions_pos = collisions_pos[0:min(self.m*2//3,len(collisions_pos))]
        #collisions_neg = collisions_neg[0:min(self.m*2//3,len(collisions_neg))]
        # sample
        r.seed(None)
        ur = r.random()
        a = 0
        z = None
        for i in range(self.t//2):
            if i in collisions or i+self.t//2 in collisions:
                if i in collisions:
                    a += np.exp(self.ep) / self.normalizer
                    if a > ur:
                        z = i
                        break
                    a += self.b / self.normalizer
                    if a > ur:
                        z = i+self.t//2
                        break
                else:
                    a += np.exp(self.ep) / self.normalizer
                    if a > ur:
                        z = i+self.t//2
                        break
                    a += self.b / self.normalizer
                    if a > ur:
                        z = i
                        break
            else:
                #assert False
                a += (self.normalizer - len(collisions) * (np.exp(self.ep)+self.b)) / (
                            (self.t - 2*len(collisions)) * self.normalizer)
                if a > ur:
                    z = i
                    break
                a += (self.normalizer - len(collisions) * (np.exp(self.ep)+self.b)) / (
                        (self.t - 2*len(collisions)) * self.normalizer)
                if a > ur:
                    z = i+self.t//2
                    break
        #print("COCO Randomizer", collisions, z, seed)
        return z, seed


    def randomizer(self, secrets, seed=None):
        xs = utils.bitarrayToList(secrets)

        tstart = time.process_time()
        z, seed = self.coreRandomizer(xs, seed)
        self.clienttime += time.process_time()-tstart

        return self.recorder(z, seed)


    def recorder(self, z, seed):
        # record a value as a hit map
        tstart = time.process_time()
        pub = np.zeros(self.d+self.m, dtype=int)
        for i in range(0, (self.d+self.m)):
            v = None
            if i < self.d//2:
                # +1
                r.seed(seed+i)
                v = r.randint(0, self.t)
                if v == z:
                    pub[i] = 1
            elif i < self.d:
                # -1
                r.seed(seed+i-self.d//2)
                v = r.randint(0, self.t)
                v = v-self.t//2 if v >= self.t//2 else v+self.t//2
                if v == z:
                    pub[i] = 1

        self.servertime += time.process_time()-tstart
        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        #print("coco hits", hits, n)
        tstart = time.process_time()
        #fs = np.array([(hits[i]-n*(nocollision[i]*1.0/self.t+(1-nocollision[i])*(self.trate+self.frate)/2))/((nocollision[i]*self.trate+(1-nocollision[i])*(self.trate+self.frate)/2)-1.0/self.t) for i in range(0, self.d+self.m)])
        #fs = np.array([(hits[i]-n*1.0/self.t)/(self.trate-1.0/self.t) for i in range(0, self.d+self.m)])
        #fs = np.array([(hits[i]-n*(nocollision[i]*1.0/self.t+(1-nocollision[i])*(self.trate+self.frate)/2))/(self.trate+self.frate-2.0/self.t) for i in range(0, self.d+self.m)])
        fs = np.array([(hits[i]-n*1.0/self.t)/(self.trate+self.frate-2.0/self.t) for i in range(0, self.d+self.m)])
        us = np.array([hits[i]/(self.trate-self.frate) for i in range(0, self.d+self.m)])
        if self.clip:
            delta = np.sqrt(self.m*np.log(n))/(1.25*self.ep)+np.sqrt(self.m*np.log(n))/(2.5)
            clip = min(1/(self.trate-self.frate), delta)
            us = np.array([hits[i]*clip for i in range(0, self.d+self.m)])
        d = self.d
        fs[0:d//2] = fs[0:d//2]+fs[d//2:d]
        fs[d//2:d] = (us[0:d//2]-us[d//2:d])/(1.0-1.0*self.conflict)
        self.servertime += time.process_time()-tstart
        #print("Conflicts", self.conflicts/(n*self.m), n)
        self.conflicts = 0
        #print("Coco", fs[d // 2:d] / n)
        return fs/n

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return (self.m*self.trate*(1.0-self.trate)+self.d*self.frate*(1-self.frate))/(n*(self.trate-self.frate)**2)
