import numpy as np
import scipy
import matplotlib.pyplot as plt
from CoverAlgorithm import *
from SimilarityFusion import *
import argparse
from pySeqAlign import swconstrained


"""====================================================
                UTILITY FUNCTIONS
===================================================="""

def get_ssm(X):
    """
    Fast code for computing the self-similarity matrix of a point cloud
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
    D = 1 - D #Make sure distance 0 is the same and distance 2 is the most different
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
        plt.plot(shiftScores)
        plt.title("OTI")
        plt.show()
    return np.argmax(shiftScores)

def get_csm_blocked_cosine_oti(X, Y, C1, C2):
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
    return get_csm_cosine(X1, Y)


def median_resize_block(X, i1, i2, frames_per_block):
    """
    Median aggregate features into a coarser list
    Parameters
    ----------
    X: ndarray(n_frames, n_feats)
        An array of features
    i1: int
        Index at beginning of block
    i2: int
        Index at end of block
    frames_per_block: int
        Number of frames to which to downsample
    """
    import librosa
    idxs = np.linspace(i1, i2, frames_per_block-1)
    idxs = np.array(np.round(idxs), dtype=int)
    res = librosa.util.sync(X.T, idxs, aggregate=np.median).T
    ret = res
    if res.shape[0] > frames_per_block:
        ret = res[0:frames_per_block, :]
    elif res.shape[0] < frames_per_block:
        ret = np.zeros((frames_per_block, res.shape[1]))
        ret[0:res.shape[0], :] = res
    return ret



"""====================================================
            FEATURE COMPUTATION/COMPARISON
===================================================="""

class EarlyFusion(CoverAlgorithm):
    """
    Attributes
    ----------
    Same as CoverAlgorithms, plus
    chroma_type: string
        Type of chroma to use (key into features)
    blocksize: int
        Number of beats to take in each block
    mfccs_per_block: int
        Resize to this number of MFCC frames in each block
    chroma_blocksize: int
        Number of beats to take in a chroma block
    chromas_per_block: int
        Resize to this number of chroma frames in each block
    kappa: float
        Neighborhood factor for binary thresholding
    K: int
        Number of nearest neighbors to use in SNF
    niters: int
        Number of iterations in SNF
    all_block_feats: dict
        A cache of features computed by load_features
    """
    def __init__(self, datapath="../features_covers80", chroma_type='hpcp', shortname='Covers80', blocksize=20, mfccs_per_block=50, ssm_res=50, chromas_per_block=40, kappa=0.1, K=10, niters=5, log_times=False):
        CoverAlgorithm.__init__(self, "EarlyFusion", datapath=datapath, shortname=shortname, \
                                similarity_types=["mfccs", "ssms", "chromas"])
        self.chroma_type = chroma_type
        self.blocksize = blocksize
        self.mfccs_per_block = mfccs_per_block
        self.chromas_per_block = chromas_per_block
        self.kappa = kappa
        self.K = K
        self.niters = niters
        self.all_block_feats = {} # Cached features
        self.log_times = log_times
        if log_times:
            self.fout = open("earlyfusionlog.txt", "w")

    def load_features(self, i, do_plot=False):
        """
        Return a dictionary of all of the beat-synchronous blocked features
        Parameters
        ----------
        i: int
            Index of the song in the corpus for which to compute features
        Returns
        -------
        block_features: dict {
            'mfccs': ndarray(n_blocks, 20*mfccs_per_block)
                Array of blocked Z-normalized raw MFCCs
            'ssms': ndarray(n_blocks, mfccs_per_block*(mfccs_per_block-1)/2)
                Upper triangular part of all SSMs for blocked
                Z-normalized MFCCs
            'chromas': ndarray(n_blocks, 12*chroma_dim)
                Array of blocked chromas
            'chroma_med': ndarray(12)
                Median of all chroma frames across song (for OTI)
        }
        """
        import librosa.util
        if i in self.all_block_feats:
            return self.all_block_feats[i]
        tic = time.time()
        block_feats = {}
        feats = CoverAlgorithm.load_features(self, i)
        chroma = feats[self.chroma_type]
        mfcc = feats['mfcc_htk'].T

        onsets = feats['madmom_features']['onsets']
        n_beats = len(onsets)
        n_blocks = n_beats - self.blocksize

        ## Step 1: Compute raw MFCC and MFCC SSM blocked features
        # Allocate space for MFCC-based features
        block_feats['mfccs'] = np.zeros((n_blocks, self.mfccs_per_block*mfcc.shape[1]), dtype=np.float32)
        pix = np.arange(self.mfccs_per_block)
        I, J = np.meshgrid(pix, pix)
        dpixels = int(self.mfccs_per_block*(self.mfccs_per_block-1)/2)
        block_feats['ssms'] = np.zeros((n_blocks, dpixels), dtype=np.float32)
        # Compute MFCC-based features
        for b in range(n_blocks):
            i1 = onsets[b]
            i2 = onsets[b+self.blocksize-1]
            x = median_resize_block(mfcc, i1, i2, self.mfccs_per_block)
            # Z-normalize
            x -= np.mean(x, 0)[None, :]
            xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
            xnorm[xnorm == 0] = 1
            xn = x / xnorm
            block_feats['mfccs'][b, :] = xn.flatten()
            # Create SSM, resize, and save
            D = get_ssm(xn)
            block_feats['ssms'][b, :] = D[I < J] # Upper triangular part
        
        ## Step 2: Compute chroma blocks
        block_feats['chromas'] = np.zeros((n_blocks, self.chromas_per_block*chroma.shape[1]), dtype=np.float32)
        block_feats['chroma_med'] = np.median(chroma, axis=0)
        for b in range(n_blocks):
            i1 = onsets[b]
            i2 = onsets[b+self.blocksize]
            x = median_resize_block(chroma, i1, i2, self.chromas_per_block)
            block_feats['chromas'][b, :] = x.flatten()
        
        ## Step 3: Precompute Ws for each features
        ssm_fns = {'chromas':lambda x: get_csm_cosine(x, x), 'mfccs':get_ssm, 'ssms':get_ssm}
        for feat in ssm_fns:
            d = ssm_fns[feat](block_feats[feat])
            block_feats['%s_W'%feat] = getW(d, self.K)
            
        self.all_block_feats[i] = block_feats # Cache features
        if self.log_times:
            self.fout.write("Features %.3g\n"%(time.time()-tic))
            self.fout.flush()
        return block_feats
    
    def similarity(self, i, j):
        print(i, j)
        feats1 = self.load_features(i)
        feats2 = self.load_features(j)
        ## Step 1: Create all of the parent SSMs
        Ws = {}
        scores = {}
        CSMs = {}
        tic = time.time()
        CSMs['mfccs'] = get_csm(feats1['mfccs'], feats2['mfccs'])
        M, N = CSMs['mfccs'].shape[0], CSMs['mfccs'].shape[1]
        scores['mfccs'] = swconstrained(CSMs['mfccs'].flatten(), M, N)
        CSMs['ssms'] = get_csm(feats1['ssms'], feats2['ssms'])
        scores['ssms'] = swconstrained(CSMs['ssms'].flatten(), M, N)
        CSMs['chromas'] = get_csm_blocked_cosine_oti(feats1['chromas'], feats2['chromas'], \
                                                    feats1['chroma_med'], feats2['chroma_med'])
        scores['chromas'] = swconstrained(CSMs['chromas'].flatten(), M, N)
        if self.log_times:
            self.fout.write("Raw: %.3g\n"%(time.time()-tic))
            self.fout.flush()

        for s in scores:
            self.Ds[s][i, j] = scores[s]
        return scores
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with Early Similarity Network Fusion of HPCP, MFCC, and MFCC SSMs",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="Covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")
    parser.add_argument("-l", '--log_times', type=int, action="store", default=0,
                        help="Whether to log times to a file")

    cmd_args = parser.parse_args()

    ef = EarlyFusion(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname, log_times=bool(cmd_args.log_times))
    ef.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)

    print("... Done ....")

