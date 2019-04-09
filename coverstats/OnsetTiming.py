"""
Purpose: To use sublevelset filtrations as an invariant to
parameterization for describing local tempo curves
"""
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import glob
from ripser import ripser
from persim import plot_diagrams
from scipy import sparse
import scipy
import scipy.io as sio
import seaborn as sns
from scipy.stats import ks_2samp
from coverstats import *

def getLowerStarFiltration(x, infinitymax=True):
    """
    Do a sublevelset filtration
    """
    N = x.size
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]
    #dgm0 = dgm0[dgm0[:, 1]-dgm0[:, 0] > 1e-3, :]
    if infinitymax:
        dgm0[np.isinf(dgm0[:, 1]), 1] = np.max(x)
    return dgm0

def getOnsetMeans(x, win=20):
    """
    Return 
    """
    x = x[1::] - x[0:-1]
    M = x.size-win+1
    X = np.zeros((M, win))
    for k in range(win):
        X[:, k] = x[k:k+M]
    return np.mean(X, 1)/np.mean(x)

def getPersistenceImage(dgm, plims, res, weightfn = lambda b, l: l, psigma = None):
    """
    Return a persistence image (Adams et al.)
    :param dgm: Nx2 array holding persistence diagram
    :param plims: An array [birthleft, birthright, lifebottom, lifetop] \
        limits of the actual grid will be rounded based on res
    :param res: Width of each pixel
    :param weightfn(b, l): A weight function as a function of birth time\
        and life time
    :param psigma: Standard deviation of each Gaussian.  By default\
        None, which indicates it should be res/2.0
    """
    #Convert to birth time/lifetime
    I = np.array(dgm)
    I[:, 1] = I[:, 1] - I[:, 0]
    
    #Create grid
    lims = np.array([np.floor(plims[0]/res), np.ceil(plims[1]/res), np.floor(plims[2]/res), np.ceil(plims[3]/res)])
    xr = np.arange(int(lims[0]), int(lims[1])+2)*res
    yr = np.arange(int(lims[2]), int(lims[3])+2)*res
    sigma = res/2.0
    if psigma:
        sigma = psigma        
            
    #Add each integrated Gaussian
    PI = np.zeros((len(yr)-1, len(xr)-1))
    for i in range(I.shape[0]):
        [x, y] = I[i, :]
        w = weightfn(x, y)
        if w == 0:
            continue
        #CDF of 2D isotropic Gaussian is separable
        xcdf = scipy.stats.norm.cdf((xr - x)/sigma)
        ycdf = scipy.stats.norm.cdf((yr - y)/sigma)
        X = ycdf[:, None]*xcdf[None, :]
        #Integral image
        PI += weightfn(x, y)*(X[1::, 1::] - X[0:-1, 1::] - X[1::, 0:-1] + X[0:-1, 0:-1])
    return {'PI':PI, 'xr':xr[0:-1], 'yr':yr[0:-1]}


if __name__ == '__main__':
    pairs = get_cover_pairs(lambda res: res['madmom_features']['onsets'])
    # Assume time series can be in the range [0, 2]
    pilims =  [0.7, 1.3, 0, 0.6]
    pilimsneg =  [-1.3, -0.7, 0, 0.6]
    pires = 0.004
    psigma = 0.03
    Is = []
    for i, pair in enumerate(pairs.keys()):
        print(i)
        for k in range(2):
            y = getOnsetMeans(pairs[pair][k])
            IUp = getLowerStarFiltration(y)
            PIUp = getPersistenceImage(IUp, pilims, pires, psigma=psigma)['PI'].flatten()
            IDown = getLowerStarFiltration(-y)
            PIDown = getPersistenceImage(IUp, pilimsneg, pires, psigma=psigma)['PI'].flatten()
            I = np.concatenate((PIUp, PIDown))
            Is.append(I)
    
    dcover = np.zeros(int(len(Is)/2))
    dfalse = np.zeros_like(dcover)
    for i in range(dcover.size):
        img1 = Is[i*2]
        img2 = Is[i*2+1]
        dcover[i] = np.sqrt(np.sum((img1-img2)**2))
        # Pick out a false cover
        idx = np.random.randint(dcover.size-1)
        if idx == i:
            idx = dcover.size-1
        img2 = Is[idx*2]
        dfalse[i] = np.sqrt(np.sum((img1-img2)**2))
    
    sio.savemat("onsettiming_%.3g"%psigma, {"dcover":dcover, "dfalse":dfalse})

    bins = np.linspace(0, np.quantile(dfalse, 0.98), 20)
    print(bins[-1])
    plt.figure(figsize=(10, 6))
    sns.distplot(dcover, kde=True, norm_hist=True, bins=bins)
    sns.distplot(dfalse, kde=True, norm_hist=True, bins=bins)
    plt.xlabel("Persitence Image Distance")
    plt.ylabel("Density")
    plt.title("Persistence Image Distances")
    plt.legend(["True Covers", "False Covers"])
    plt.savefig("OnsetTimings_%.3g.svg"%psigma, bbox_inches='tight')
    print(np.mean(dcover))
    print(np.mean(dfalse))
    print(ks_2samp(dcover, dfalse))
        
