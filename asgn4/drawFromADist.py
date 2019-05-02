import numpy as np

def drawFromADist(p):
    if (np.sum(p)==0):
        p = 0.05*np.ones(p.shape[0])

    p = p/np.sum(p)
    c = np.cumsum(p)
    rand = np.random.uniform()
    idx = (rand < c)
    sample = np.arange(p.shape[0])
    sample = sample[idx]
    sample = sample[0]
    out = np.zeros(p.shape[0])
    out[sample] = 1.0

    return sample
