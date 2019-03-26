import numpy as np
import scipy
import matplotlib.pyplot as plt
from CoverAlgorithm import *
import argparse


def chrompwr(X, P=.5):
    """
    Y = chrompwr(X,P)  raise chroma columns to a power, preserving norm
    2006-07-12 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = X.shape
    # norms of each input col
    CMn = np.tile(np.sqrt(np.sum(X * X, axis=0)), (nchr, 1))
    CMn[np.where(CMn==0)] = 1
    # normalize each input col, raise to power
    CMp = np.power(X/CMn, P)
    # norms of each resulant column
    CMpn = np.tile(np.sqrt(np.sum(CMp * CMp, axis=0)), (nchr, 1))
    CMpn[np.where(CMpn==0)] = 1.
    # rescale cols so norm of output cols match norms of input cols
    return CMn * (CMp / CMpn)


def btchroma_to_fftmat(btchroma, win=75):
    """
    Stack the flattened result of fft2 on patches 12 x win
    Translation of my own matlab function
    -> python: TBM, 2011-11-05, TESTED
    """
    # 12 semitones
    nchrm, nbeats = btchroma.shape
    assert nchrm == 12, 'beat-aligned matrix transposed?'
    if nbeats < win:
        return None
    # output
    fftmat = np.zeros((nchrm * win, nbeats - win + 1))
    for i in range(nbeats-win+1):
        F = scipy.fftpack.fft2(btchroma[:,i:i+win])
        F = np.sqrt(np.real(F)**2 + np.imag(F)**2)
        patch = scipy.fftpack.fftshift(F)
        fftmat[:, i] = patch.flatten()
    return fftmat


class FTM2D(CoverAlgorithm):
    """
    Attributes
    ----------
    Same as CoverAlgorithms, plus
    shingles: {int: ndarray(WIN*chromabins)}
        A map from the song index to the FFT2DM shingles, so that
        they are cached
    chroma_type: string
        Type of chroma to use (key into features)
    """
    def __init__(self, datapath="../features_covers80", chroma_type='hpcp', PWR=1.96, WIN=75, C=5):
        CoverAlgorithm.__init__(self, "FTM2D", datapath)
        self.PWR = PWR
        self.WIN = WIN
        self.C = C
        self.chroma_type = chroma_type
        self.shingles = {}

    def load_features(self, i, do_plot=False):
        if i in self.shingles:
            # If the result has already been cached, return the cache
            return self.shingles[i]
        # Otherwise, compute the shingle
        import librosa.util
        feats = CoverAlgorithm.load_features(self, i)
        hpcp_orig = feats[self.chroma_type].T
        # Synchronize HPCP to the beats
        onsets = feats['madmom_features']['onsets']
        hpcp = librosa.util.sync(hpcp_orig, onsets, aggregate=np.median)

        chroma = chrompwr(hpcp, self.PWR)
        # Get all 2D FFT magnitude shingles
        shingles = btchroma_to_fftmat(chroma, self.WIN).T
        Norm = np.sqrt(np.sum(shingles**2, 1))
        Norm[Norm == 0] = 1
        shingles = np.log(self.C*shingles/Norm[:, None] + 1)
        shingle = np.median(shingles, 0) # Median aggregate
        shingle = shingle/np.sqrt(np.sum(shingle**2))

        if do_plot:
            import librosa.display
            plt.subplot(311)
            librosa.display.specshow(librosa.amplitude_to_db(hpcp_orig, ref=np.max))
            plt.title("Original")
            plt.subplot(312)
            librosa.display.specshow(librosa.amplitude_to_db(hpcp, ref=np.max))
            plt.title("Beat-synchronous Median Aggregated")
            plt.subplot(313)
            plt.imshow(np.reshape(shingle, (hpcp.shape[0], self.WIN)))
            plt.title("Median FFT2D Mag Shingle")
            plt.show()
        self.shingles[i] = shingle
        return shingle
    
    def similarity(self, i, j):

        s1 = self.load_features(i)
        s2 = self.load_features(j)
        dSqr = np.sum((s1-s2)**2)
        # Since similarity should be high for two things
        # with a small distance, take the negative exponential
        sim = np.exp(-dSqr)
        self.D[i, j] = sim
        self.D[j, i] = sim
        return sim


def ftm2d_allpairwise(datapath='../data/features_covers80',
                      chroma_type='hpcp',
                      parallel=0,
                      n_cores=12,
                      cached=False):
    """
    Show how one might do all pairwise comparisons between songs,
    with code that is amenable to parallelizations.
    This will go slowly at the beginning but then it will speed way
    up because it caches the shingles for each song
    """
    from itertools import combinations
    import scipy.io as sio
    ftm = FTM2D(datapath=datapath, chroma_type=chroma_type)

    if cached:
        D = sio.loadmat("FTM2D_{}.mat")["D"]
        ftm.D = D
        ftm.get_all_clique_ids()
    else:
        if parallel == 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_cores, verbose=1)(
                delayed(ftm.similarity)(i, j) for idx, (i, j) in enumerate(combinations(range(len(ftm.filepaths)), 2)))
            ftm.D += ftm.D.T
            sio.savemat("FTM2D.mat", {"D": ftm.D})
            ftm.get_all_clique_ids() # Since nothing has been cached
        else:
            for idx, (i, j) in enumerate(combinations(range(len(ftm.filepaths)), 2)):
                ftm.similarity(i, j)
                if idx%100 == 0:
                    print((i, j))
            ftm.D += ftm.D.T
            sio.savemat("FTM2D.mat", {"D":ftm.D})
    ftm.getEvalStatistics()
    if parallel == 1:
        import shutil
        try:
            shutil.rmtree('d_mat')
        except:  # noqa
            print('Could not clean-up automatically.')


def ftm2d_allpairwise_covers80(chroma_type='hpcp'):
    """
    Show how one might do all pairwise comparisons between songs,
    with code that is amenable to parallelizations.
    This will go slowly at the beginning but then it will speed way
    up because it caches the shingles for each song
    """
    from itertools import combinations
    import scipy.io as sio
    ftm = FTM2D(datapath="features_covers80", chroma_type=chroma_type)
    for idx, (i, j) in enumerate(combinations(range(len(ftm.filepaths)), 2)):
        ftm.similarity(i, j)
        if idx%100 == 0:
            print((i, j))
    
    ftm.getEvalStatistics()


if __name__ == '__main__':
    #ftm2d_allpairwise_covers80(chroma_type='crema')
    parser = argparse.ArgumentParser(description="Benchmarking with 2D Fourier Transform Magnitude Coefficients",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")

    cmd_args = parser.parse_args()

    ftm2d_allpairwise(cmd_args.datapath, cmd_args.chroma_type, cmd_args.parallel, cmd_args.n_cores)

    print("... Done ....")

