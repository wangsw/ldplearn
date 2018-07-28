__author__ = 'wangsw'
# util functions

import math
import numpy as np
import scipy as sp
import numpy.random as r

def binarysearch(l, v):
    # search the corresponding scope holding v
    s = 0
    e = len(l)-1
    while v < l[math.floor((s+e)/2)] or v >= l[math.floor((s+e)/2) + 1]:
        if v < l[math.floor((s+e)/2)]:
            e = math.floor((s+e)/2)
        else:
            s = math.floor((s+e)/2)
    return math.floor((s+e)/2)


def reservoirsample(l, m):
    # sample m elements from list l
    samples = l[0:m]
    for i in range(m, len(l)):
        index = r.randint(0, i+1)
        if index < m:
            samples[index] = l[i]
    return samples


def recorder(hits, pub):
    # record pub to hits
    for s in pub:
        hits[s] += 1
    return hits


def histogramer(d, n, dist=None):
    # create a size n histogram according to distribution dist
    if dist == None:
        dist = [1/d]*d
    h = np.full((d,), 0, dtype=int)
    for i in range(0, d):
        h[i] = int(r.binomial(n, dist[i]))
    while np.sum(h) != n:
        diff = n-np.sum(h)
        ri = r.randint(0, d-1)
        if h[ri]+diff >= 0:
            h[ri] = h[ri]+diff
        else:
            h[ri] = 0
    #print(h)
    return h


def projector(od):
    # project od to probability simplex
    u = -np.sort(-od)
    #print("sorted:\t", u)
    sod = np.zeros(len(od))
    sod[0] = u[0]
    for i in range(1, len(od)):
        sod[i] = sod[i-1]+u[i]

    for i in range(0, len(od)):
        sod[i] = u[i]+(1.0-sod[i])/(i+1)

    p = 0
    for i in range(len(od)-1, -1, -1):
        if sod[i] > 0.0:
            p = i
            break

    q = sod[p]-u[p]

    x = np.zeros(len(od))
    for i in range(0, len(od)):
        x[i] = np.max([od[i]+q, 0.0])
    #print("projected:\t",x)
    return x




def distributor(n, histogram, mechanism):
    # randomize items in the hitogram and return observed hits
    hits = np.full(len(histogram), 0, dtype=int)
    for i in range(0, len(histogram)):
        for c in range(0, histogram[i]):
            recorder(hits, mechanism.randomizer(i))
    return mechanism.decoder(hits, n)



def paddingSets(sets, n, d, m):
    for i in range(0, n):
        if np.sum(sets[i]) > m:
            values = [j for j in range(0, d) if sets[i,j] == 1]
            valuesnow = utils.reservoirsample(values, m)
            for j in range(0, d):
                if j in valuesnow:
                    sets[i,j] = 1
                else:
                    sets[i,j] = 0
        else:
            for j in range(d, d + m - sum(sets[i])):
                sets[i,j] = 1
    return sets


def randomSets(n, d, m, dist=None):
    sets = np.zeros((n, d+m), dtype=int)
    if dist == None:
        dist = np.ones(d, dtype=float)
        dist = m*dist/d

    for i in range(0, n):
        subset = np.full(d+m, 0, dtype=int)
        for j in range(0, d):
            p = r.random(1)
            if p < dist[j]:
                subset[j] = 1
        sets[i] = subset
    return sets



def countsToSizes(dl):
    # compute m-size:probability pairs from estimated padding items
    sizes = np.full(len(dl)+1, 0, dtype=float)
    sizes[0] = 1.0 - dl[0]
    for i in range(0, len(dl)-1):
        sizes[i+1] = dl[i] - dl[i+1]
    sizes[len(dl)] = dl[-1]
    #print(sizes.tolist())
    return sizes
