"""
Programmer: Chris Tralie, 12/2016 (ctralie@alumni.princeton.edu)
Purpose: To implement similarity network fusion approach described in
[1] Wang, Bo, et al. "Unsupervised metric fusion by cross diffusion." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
[2] Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." Nature methods 11.3 (2014): 333-337.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.io as sio
import time
import os

def getW(D, K, Mu = 0.5):
    """
    Return affinity matrix
    :param D: Self-similarity matrix
    :param K: Number of nearest neighbors
    :param Mu: Nearest neighbor hyperparameter (default 0.5)
    """
    #W(i, j) = exp(-Dij^2/(mu*epsij))
    DSym = 0.5*(D + D.T)
    np.fill_diagonal(DSym, 0)

    Neighbs = np.partition(DSym, K+1, 1)[:, 0:K+1]
    MeanDist = np.mean(Neighbs, 1)*float(K+1)/float(K) #Need this scaling
    #to exclude diagonal element in mean
    #Equation 1 in SNF paper [2] for estimating local neighborhood radii
    #by looking at k nearest neighbors, not including point itself
    Eps = MeanDist[:, None] + MeanDist[None, :] + DSym
    Eps = Eps/3
    Denom = (2*(Mu*Eps)**2)
    Denom[Denom == 0] = 1
    W = np.exp(-DSym**2/Denom)
    return W

def getWCSM(CSMAB, k1, k2, Mu = 0.5):
    """
    Get a cross similarity matrix from a cross dissimilarity matrix
    :param CSMAB: Cross-similarity matrix
    :param k1: Number of neighbors across rows
    :param k2: Number of neighbors down columns
    :param Mu: Nearest neighbor hyperparameter
    :returns W: Exponential weighted similarity matrix
    """
    Neighbs1 = np.partition(CSMAB, k2, 1)[:, 0:k2]
    MeanDist1 = np.mean(Neighbs1, 1)

    Neighbs2 = np.partition(CSMAB, k1, 0)[0:k1, :]
    MeanDist2 = np.mean(Neighbs2, 0)
    Eps = MeanDist1[:, None] + MeanDist2[None, :] + CSMAB
    Eps /= 3
    return np.exp(-CSMAB**2/(2*(Mu*Eps)**2))

def setupWCSMSSM(WSSMA, WSSMB, WCSMAB):
    """
    Get the following kernel cross-similarity matrix
                [ WSSMA      WCSMAB ]
                [ WCSMBA^T   WSSMB  ]
    :param WSSMA: W matrix for upper left SSM part
    :param WSSMB: W matrix for lower SSM part
    :param WCSMAB: Cross-similarity part
    :returns: Matrix with them all together
    """

    M = WSSMA.shape[0]
    N = WSSMB.shape[0]
    W = np.zeros((N+M, N+M))
    W[0:M, 0:M] = WSSMA
    W[0:M, M::] = WCSMAB
    W[M::, 0:M] = WCSMAB.T
    W[M::, M::] = WSSMB
    return W

def getWCSMSSM(SSMA, SSMB, CSMAB, K, Mu = 0.5):
    """
    Cross-Affinity Matrix.  Do a special weighting of nearest neighbors
    so that there are a proportional number of similarity neighbors
    and cross neighbors
    :param SSMA: MxM self-similarity matrix for signal A
    :param SSMB: NxN self-similarity matrix for signal B
    :param CSMAB: MxN cross-similarity matrix between A and B
    :param K: Total number of nearest neighbors per row used
        to tune exponential threshold
    :param Mu: Hyperparameter for nearest neighbors
    :return W: Parent W matrix
    """
    M = SSMA.shape[0]
    N = SSMB.shape[0]
    #Split the neighbors evenly between the CSM
    #and SSM parts of each row
    k1 = int(K*float(M)/(M+N))
    k2 = K - k1

    WSSMA = getW(SSMA, k1, Mu)
    WSSMB = getW(SSMB, k2, Mu)
    WCSMAB = getWCSM(CSMAB, k1, k2, Mu)
    return setupWCSMSSM(WSSMA, WSSMB, WCSMAB)

def getP(W, diagRegularize = False):
    """
    Turn a similarity matrix into a proability matrix,
    with each row sum normalized to 1
    :param W: (MxM) Similarity matrix
    :param diagRegularize: Whether or not to regularize
    the diagonal of this matrix
    :returns P: (MxM) Probability matrix
    """
    if diagRegularize:
        P = 0.5*np.eye(W.shape[0])
        WNoDiag = np.array(W)
        np.fill_diagonal(WNoDiag, 0)
        RowSum = np.sum(WNoDiag, 1)
        RowSum[RowSum == 0] = 1
        P = P + 0.5*WNoDiag/RowSum[:, None]
        return P
    else:
        RowSum = np.sum(W, 1)
        RowSum[RowSum == 0] = 1
        P = W/RowSum[:, None]
        return P

def getS(W, K):
    """
    Same thing as P but restricted to K nearest neighbors
        only (using partitions for fast nearest neighbor sets)
    (**note that nearest neighbors here include the element itself)
    :param W: (MxM) similarity matrix
    :param K: Number of neighbors to use per row
    :returns S: (MxM) S matrix, J: (MxK) matrix of nearest neighbor indices
    """
    N = W.shape[0]
    J = np.argpartition(-W, K, 1)[:, 0:K]
    I = np.tile(np.arange(N)[:, None], (1, K))
    V = W[I.flatten(), J.flatten()]
    #Now figure out L1 norm of each row
    V = np.reshape(V, J.shape)
    SNorm = np.sum(V, 1)
    SNorm[SNorm == 0] = 1
    V = V/SNorm[:, None]
    [I, J, V] = [I.flatten(), J.flatten(), V.flatten()]
    S = sparse.coo_matrix((V, (I, J.flatten())), shape=(N, N)).tocsr()
    return S, J


def doSimilarityFusionWs(Ws, K = 5, niters = 20, reg_diag = 1):
    """
    Perform similarity fusion between a set of exponentially
    weighted similarity matrices
    :param Ws: An array of NxN affinity matrices for N songs
    :param K: Number of nearest neighbors
    :param niters: Number of iterations
    :param reg_diag: Identity matrix regularization parameter for
        self-similarity promotion
    :return D: A fused NxN similarity matrix
    """
    tic = time.time()
    #Full probability matrices
    Ps = [getP(W) for W in Ws]
    #Nearest neighbor truncated matrices
    Ss = [getS(W, K)[0] for W in Ws]

    #Now do cross-diffusion iterations
    Pts = [np.array(P) for P in Ps]
    nextPts = [np.zeros(P.shape) for P in Pts]

    N = len(Pts)
    for it in range(niters):
        for i in range(N):
            nextPts[i] *= 0
            tic = time.time()
            for k in range(N):
                if i == k:
                    continue
                nextPts[i] += Pts[k]
            nextPts[i] /= float(N-1)
            #Need S*P*S^T, but have to multiply sparse matrix on the left
            tic = time.time()
            A = Ss[i].dot(nextPts[i].T)
            nextPts[i] = Ss[i].dot(A.T)
            if reg_diag > 0:
                nextPts[i] += reg_diag*np.eye(nextPts[i].shape[0])
        Pts = nextPts
    FusedScores = np.zeros(Pts[0].shape)
    for Pt in Pts:
        FusedScores += Pt
    return FusedScores/N

def doSimilarityFusion(Scores, K = 10, niters = 10, reg_diag = 1):
    """
    Do similarity fusion on a set of NxN distance matrices.
    Parameters the same as doSimilarityFusionWs
    :returns (An array of similarity matrices for each feature, Fused Similarity Matrix)
    """
    #Affinity matrices
    Ws = [getW(D, K) for D in Scores]
    return (Ws, doSimilarityFusionWs(Ws, K, niters, reg_diag))

def testSimilarityFusion():
    """
    Test similarity fusion on SSMs from one of the
    songs in the covers80 dataset
    """
    from EarlyFusion import EarlyFusion, get_ssm, get_csm_blocked_cosine_oti
    ef = EarlyFusion()
    tic = time.time()
    features = ef.load_features(2)
    print("Elapsed Time Block Features: %.3g"%(time.time()-tic))
    D1 = get_ssm(features['mfccs'])
    Chroma = features['chromas']
    CMed = features['chroma_med']
    D2 = get_csm_blocked_cosine_oti(Chroma, Chroma, CMed, CMed)
    N = min(D1.shape[0], D2.shape[0])
    D1 = D1[0:N, 0:N]
    D2 = D2[0:N, 0:N]

    Ws, DFPython = doSimilarityFusion([D1, D2])
    np.fill_diagonal(DFPython, 0)
    
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(D1)
    plt.title("MFCCs")
    plt.subplot(132)
    plt.imshow(D2)
    plt.title("Chromas")
    plt.subplot(133)
    plt.imshow(DFPython)
    plt.title("Fused")
    plt.show()

if __name__ == '__main__':
    testSimilarityFusion()