# -*- coding: utf-8 -*-
"""
@2019
"""
import argparse
import librosa

try:
    from pySeqAlign import qmax, dmax
except ImportError:
    raise ImportError("Cannot import pySeqAlign cython module.")
from .algorithm_template import CoverAlgorithm
from .utils.cross_recurrence import *
from .utils.similarity_fusion import *


def global_chroma(chroma):
    """Computes global chroma of a input chroma vector"""
    if chroma.shape[1] not in [12, 24, 36]:
        raise IOError("Wrong axis for the input chroma array. Expected shape '(frame_size, bin_size)'")
    return np.divide(chroma.sum(axis=0), np.max(chroma.sum(axis=0)))


class ChenFusion(CoverAlgorithm):
    """
    Attributes
    ----------
    Same as CoverAlgorithms, plus
    chroma_type: string
        Type of chroma to use (key into features)
    downsample_fac: int
        The factor by which to downsample the HPCPs with
        median aggregation
    all_feats: {int: dictionary}
        Cached features
    """

    def __init__(self, dataset_csv, datapath="../features_covers80", chroma_type='hpcp', shortname='benchmark',
                 oti=True, kappa=0.095, tau=1, m=9, downsample_fac=40):
        self.oti = oti
        self.tau = tau
        self.m = m
        self.chroma_type = chroma_type
        self.kappa = kappa
        self.tau = tau
        self.m = m
        self.downsample_fac = downsample_fac
        self.all_feats = {}  # For caching features (global chroma and stacked chroma)
        CoverAlgorithm.__init__(self, dataset_csv, name="ChenFusion", similarity_types=["qmax", "dmax"],
                                datapath=datapath, shortname=shortname)

    def load_features(self, i):
        if not i in self.all_feats:
            feats = CoverAlgorithm.load_features(self, i)
            # First compute global chroma (used for OTI later)
            chroma = feats[self.chroma_type]
            gchroma = global_chroma(chroma)
            # Now downsample the chromas using median aggregation
            chroma = librosa.util.sync(chroma.T, np.arange(0, chroma.shape[0], self.downsample_fac),
                                       aggregate=np.median)
            # Finally, do a stacked delay embedding
            stacked = librosa.feature.stack_memory(chroma, self.tau, self.m).T
            feats = {'gchroma': gchroma, 'stacked': stacked}
            self.all_feats[i] = feats
        return self.all_feats[i]

    def similarity(self, idxs):
        for i, j in zip(idxs[:, 0], idxs[:, 1]):
            Si = self.load_features(i)
            Sj = self.load_features(j)
            csm = get_csm_blocked_oti(Si['stacked'], Sj['stacked'], Si['gchroma'], Sj['gchroma'], get_csm_euclidean)
            csm = csm_to_binary(csm, self.kappa)
            M, N = csm.shape[0], csm.shape[1]
            D = np.zeros(M * N, dtype=np.float32)
            self.Ds["qmax"][i, j] = qmax(csm.flatten(), D, M, N)
            D *= 0
            self.Ds["dmax"][i, j] = dmax(csm.flatten(), D, M, N)

    def normalize_by_length(self):
        """
        Do a non-symmetric normalization by length
        """
        N = len(self.filepaths)
        for j in range(N):
            f = self.load_features(j)
            norm_fac = np.sqrt(f['stacked'].shape[0])
            for i in range(N):
                for key in self.Ds:
                    self.Ds[key][i, j] = norm_fac / self.Ds[key][i, j]

    def do_late_fusion(self):
        DLate = doSimilarityFusion([self.Ds[s] for s in self.Ds], K=20, niters=20, reg_diag=1)[1]
        for key in self.Ds:
            self.Ds[key] *= -1  # Switch back to larger scores being closer
        self.Ds["Late"] = DLate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with Joan Serra's Cover id algorithm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '--dataset_csv', type=str, action="store",
                        help="Input dataset csv file")
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="Covers80",
                        help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")

    cmd_args = parser.parse_args()

    chenFusion = ChenFusion(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname)
    chenFusion.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
    chenFusion.normalize_by_length()
    chenFusion.do_late_fusion()
    for similarity_type in chenFusion.Ds.keys():
        print(similarity_type)
        chenFusion.getEvalStatistics(similarity_type)
    chenFusion.cleanup_memmap()
    print("... Done ....")
