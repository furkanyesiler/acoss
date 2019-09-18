# -*- coding: utf-8 -*-
"""
@2019
"""
import numpy as np
import argparse
import scipy
from .algorithm_template import CoverAlgorithm


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
    def __init__(self, dataset_csv, datapath="../features_covers80", chroma_type='hpcp', shortname='Covers80', PWR=1.96, WIN=75, C=5):
        self.PWR = PWR
        self.WIN = WIN
        self.C = C
        self.chroma_type = chroma_type
        self.shingles = {}
        CoverAlgorithm.__init__(self, dataset_csv=dataset_csv, name="FTM2D", datapath=datapath, shortname=shortname)

    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s_%s"%(self.cachedir, self.name, self.shortname, self.chroma_type)

    def load_features(self, i, do_plot=False):
        filepath = "%s_%i.h5"%(self.get_cacheprefix(), i)
        if i in self.shingles:
            # If the result has already been cached in memory, 
            # return the cache
            return self.shingles[i]
        elif os.path.exists(filepath):
            # If the result has already been cached on disk, 
            # load it, save it in memory, and return
            self.shingles[i] = dd.io.load(filepath)['shingle']
            # Make sure to also load clique info as a side effect
            feats = CoverAlgorithm.load_features(self, i)
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
            import matplotlib.pyplot as plt
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
        dd.io.save(filepath, {'shingle':shingle})
        return shingle
    
    def similarity(self, idxs):
        (a, b) = idxs.shape
        for k in range(a):
            i = idxs[k][0]
            j = idxs[k][1]

            s1 = self.load_features(i)
            s2 = self.load_features(j)
            dSqr = np.sum((s1-s2)**2)
            # Since similarity should be high for two things
            # with a small distance, take the negative exponential
            sim = np.exp(-dSqr)
            self.Ds['main'][i, j] = sim


def chrompwr(X, P=.5):
    """
    Y = chrompwr(X,P)  raise chroma columns to a power, preserving norm
    2006-07-12 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = X.shape
    # norms of each input col
    CMn = np.tile(np.sqrt(np.sum(X * X, axis=0)), (nchr, 1))
    CMn[CMn == 0] = 1
    # normalize each input col, raise to power
    res = X/CMn
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


if __name__ == '__main__':
    # ftm2d_allpairwise_covers80(chroma_type='crema')
    parser = argparse.ArgumentParser(description="Benchmarking with 2D Fourier Transform Magnitude Coefficients",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '--dataset_csv', type=str, action="store",
                        help="Input dataset csv file")
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="Covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")

    cmd_args = parser.parse_args()

    ftm2d = FTM2D(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname)
    for i in range(len(ftm2d.filepaths)):
        ftm2d.load_features(i)
    print('Feature loading done.')
    ftm2d.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
    for similarity_type in ftm2d.Ds.keys():
        ftm2d.getEvalStatistics(similarity_type)
    ftm2d.cleanup_memmap()

    print("... Done ....")

