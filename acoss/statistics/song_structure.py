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
import scipy.linalg as sclinalg
from statistics import *
import librosa
import skimage
import time
import glob

from ..benchmark.utils.similarity_fusion import *
from ..benchmark.utils.cross_recurrence import *


def getRandomWalkLaplacianEigsDense(W, neigs):
    """
    Get eigenvectors of the random walk Laplacian by solving
    the generalized eigenvalue problem
    L*u = lam*D*u
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    
    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    D = np.diag(np.sum(W, 1))
    L = D - W
    return sclinalg.eigh(L, D, turbo=True, eigvals=(0, neigs))

def getShapeDNA(features, do_plot=False, neigs=30):
    downsample_fac = 10
    m = 20
    dim = 256
    
    chroma = features['hpcp']
    mfcc = features['mfcc_htk']

    chroma = librosa.util.sync(chroma.T, np.arange(0, chroma.shape[0], downsample_fac), aggregate=np.median)
    # Finally, do a stacked delay embedding
    chroma = librosa.feature.stack_memory(chroma, n_steps=m).T
    DChroma = get_ssm(chroma)

    mfcc[np.isinf(mfcc)] = 0
    mfcc[np.isnan(mfcc)] = 0
    mfcc = librosa.util.sync(mfcc, np.arange(0, mfcc.shape[1], downsample_fac), aggregate=np.median)
    mfcc = librosa.feature.stack_memory(mfcc, n_steps=m).T
    mfcc[np.isinf(mfcc)] = 0
    mfcc[np.isnan(mfcc)] = 0
    DMFCC = get_ssm(mfcc)

    N = min(DChroma.shape[0], DMFCC.shape[0])
    DChroma = DChroma[0:N, 0:N]
    DMFCC = DMFCC[0:N, 0:N]

    K = int(np.round(N*0.01))
    Ws, DFused = doSimilarityFusion([DChroma, DMFCC], K=K, niters=5)
    
    W = skimage.transform.resize(DFused, (dim, dim), anti_aliasing=True, mode='constant')
    w, v = getRandomWalkLaplacianEigsDense(W, neigs)
    return {'w':w, 'v':v, 'DFused':DFused, 'W':W}

def getPairs():
    import os
    if not os.path.exists("pairs.txt"):
        fout = open("pairs.txt", "w")
        files = glob.glob("feature_whatisacover/*.h5")
        pairs = {}
        for f in files:
            res = dd.io.load(f)
            label = res['label']
            if not label in pairs:
                pairs[label] = []
            pairs[label].append(f)
        for p in pairs:
            fout.write("%s %s\n"%(pairs[p][0], pairs[p][1]))
        fout.close()
    fin = open("pairs.txt")
    pairs = []
    for s in fin.readlines():
        s.strip()
        f1, f2 = s.split()
        pairs.append([f1, f2])
    return pairs

def computePairs():
    np.warnings.filterwarnings('ignore')
    pairs = getPairs()
    plt.figure(figsize=(12, 8))
    ws1 = []
    ws2 = []
    tic = time.time()
    for i, [p1, p2] in enumerate(pairs):
        res1 = getShapeDNA(dd.io.load(p1))
        res2 = getShapeDNA(dd.io.load(p2))
        ws1.append(res1['w'])
        ws2.append(res2['w'])
        if i%100 == 0:
            print("Elapsed Time %i: %.3g"%(i, time.time()-tic))
            sio.savemat("allws.mat", {'ws1':np.array(ws1), 'ws2':np.array(ws2)})

def analyzeResults():
    np.warnings.filterwarnings('ignore')
    ws1 = sio.loadmat("allws1.mat")["allws"]
    ws2 = sio.loadmat("allws2.mat")["allws"]
    N = min(ws2.shape[0], ws1.shape[0])
    ws1 = ws1[0:N, :]
    ws2 = ws2[0:N, :]
    print("N = %i"%N)
    norm = False
    k = ws2.shape[1]
    ws1 = ws1[:, 0:k]
    ws2 = ws2[:, 0:k]
    if norm:
        ws1norm = np.sqrt(np.sum(ws1**2, 1))
        ws1 /= ws1norm[:, None]
        ws2norm = np.sqrt(np.sum(ws2**2, 1))
        ws2 /= ws2norm[:, None]
    D = get_csm(ws1, ws2)
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    dcover = np.diag(D)
    dfalse = D[np.abs(I-J) > 0]
    print(ks_2samp(dcover, dfalse))
    plt.figure(figsize=(6, 3))
    bins = np.linspace(0, 3, 30)
    sns.distplot(dcover, kde=True, norm_hist=True, bins=bins)
    sns.distplot(dfalse, kde=True, norm_hist=True, bins=bins)
    plt.xlim([0, 2.5])
    plt.xlabel("Shape DNA Distance")
    plt.ylabel("Density")
    plt.title("Shape DNA Distances for Structure Comparison")
    plt.legend(["True Covers", "False Covers"])
    plt.savefig("ShapeDNA.svg", bbox_inches='tight')


def PaperFigure():
    pairs = getPairs()
    plt.figure(figsize=(6, 6))
    print(pairs[3])
    print(pairs[4][0])
    res1 = getShapeDNA(dd.io.load(pairs[3][1]))
    res2 = getShapeDNA(dd.io.load(pairs[3][0]))
    res3 = getShapeDNA(dd.io.load(pairs[4][0]))

    plt.subplot(221)
    plt.imshow(np.log(res1['W']/np.quantile(res1['W'], 0.05)), cmap='magma_r')
    plt.title('Original')
    plt.xlabel("Time")
    plt.ylabel("Time")
    plt.subplot(223)
    plt.imshow(np.log(res2['W']/np.quantile(res2['W'], 0.05)), cmap='magma_r')
    plt.title('Cover')
    plt.xlabel("Time")
    plt.ylabel("Time")
    plt.subplot(222)
    plt.imshow(np.log(res3['W']/np.quantile(res3['W'], 0.05)), cmap='magma_r')
    plt.title("Non-Cover")
    plt.xlabel("Time")
    plt.ylabel("Time")
    plt.subplot(224)
    plt.plot(res1['w'])
    plt.plot(res2['w'])
    plt.plot(res3['w'])
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.title("Shape DNA")
    plt.legend(["Original", "Cover", "Non-Cover"])
    plt.tight_layout()
    plt.savefig("ShapeDNAExample.svg", bbox_inches='tight')

if __name__ == '__main__':
    analyzeResults()
    #PaperFigure()