"""
Utility functions for cross-recurrence plots and
optimal transposition indexes
"""
import numpy as np
from scipy import sparse


def get_ssm(X):
    """
    Fast code for computing the Euclidean self-similarity
    matrix of a point cloud
    Parameters
    ----------
    X: ndarray(N, d)
        A point cloud with N points in d dimensions
    Returns
    -------
    d: ndarray(N, N)
        All pairs similarity matrix
    """
    XSqr = np.sum(X**2, 1)
    DSqr = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    DSqr[DSqr < 0] = 0
    np.fill_diagonal(DSqr, 0)
    return np.sqrt(DSqr)


def get_csm(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    Parameters
    ----------
    X: ndarray(M, d)
        A point cloud with M points in d dimensions
    Y: ndarray(N, d)
        A point cloud with N points in d dimensions
    Returns
    -------
    D: ndarray(M, N)
        An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)


get_csm_euclidean = get_csm


def get_csm_cosine(X, Y):
    """
    Return the cosine distance between all vectors in X
    and all vectors in Y
    X: ndarray(M, d)
        A point cloud with M points in d dimensions
    Y: ndarray(N, d)
        A point cloud with N points in d dimensions
    Returns
    -------
    D: ndarray(M, N)
        A cosine cross-similarity matrix
    """
    XNorm = np.sqrt(np.sum(X**2, 1))
    XNorm[XNorm == 0] = 1
    YNorm = np.sqrt(np.sum(Y**2, 1))
    YNorm[YNorm == 0] = 1
    D = (X/XNorm[:, None]).dot((Y/YNorm[:, None]).T)
    D = 1 - D # Make sure distance 0 is the same and distance 2 is the most different
    return D


def get_oti(C1, C2, do_plot = False):
    """
    Get the optimial transposition of the first chroma vector
    with respect to the second one
    Parameters
    ----------
    C1: ndarray(n_chroma_bins)
        Chroma vector 1
    C2: ndarray(n_chroma_bins)
        Chroma vector 2
    do_plot: boolean
        Plot the agreements over all shifts
    Returns
    -------
    oit: int
        An index by which to rotate the first chroma vector
        to match with the second
    """
    NChroma = len(C1)
    shiftScores = np.zeros(NChroma)
    for i in range(NChroma):
        shiftScores[i] = np.sum(np.roll(C1, i)*C2)
    if do_plot:
        import matplotlib.pyplot as plt
        plt.plot(shiftScores)
        plt.title("OTI")
        plt.show()
    return np.argmax(shiftScores)


def get_csm_blocked_oti(X, Y, C1, C2, csm_fn):
    """
    Get the cosine distance between each row of X
    and each row of Y after doing a global optimal
    transposition change from X to Y
    Parameters
    ----------
    X: ndarray(M, n_chroma_bins*chromas_per_block)
        Chroma blocks from first song
    Y: ndarray(N, n_chroma_bins*chromas_per_block)
        Chroma blocks from the second song
    C1: ndarray(n_chroma_bins)
        Global chroma vector for song 1
    C2: ndarray(n_chroma_bins)
        Global chroma vector for song 2
    csm_fn:
        Cross-similarity function to use after OTI
    Returns
    -------
    D: ndarray(M, N)
        An MxN cross-similarity matrix
    """
    NChromaBins = len(C1)
    ChromasPerBlock = int(X.shape[1]/NChromaBins)
    oti = get_oti(C1, C2)
    X1 = np.reshape(X, (X.shape[0], ChromasPerBlock, NChromaBins))
    X1 = np.roll(X1, oti, axis=2)
    X1 = np.reshape(X1, [X.shape[0], ChromasPerBlock*NChromaBins])
    return csm_fn(X1, Y)


def csm_to_binary(D, kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix, using partitions instead of
    nearest neighbors for speed
    :param D: M x N cross-similarity matrix
    :param kappa:
        If kappa = 0, take all neighbors
        If kappa < 1 it is the fraction of mutual neighbors to consider
        Otherwise kappa is the number of mutual neighbors to consider
    :returns B: MxN binary cross-similarity matrix
    """
    N = D.shape[0]
    M = D.shape[1]
    if kappa == 0:
        return np.ones_like(D)
    elif kappa < 1:
        NNeighbs = int(np.round(kappa*M))
    else:
        NNeighbs = kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M), dtype=np.uint8)
    return ret.toarray()

