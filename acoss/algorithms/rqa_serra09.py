# -*- coding: utf-8 -*-
"""
Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.
New Journal of Physics, 11(9), 093017.
"""
import argparse
import sys

from essentia.standard import ChromaCrossSimilarity, CoverSongSimilarity
from librosa.util import sync

from .algorithm_template import CoverAlgorithm
from .utils.cross_recurrence import *

__all__ = ['Serra09']


class Serra09(CoverAlgorithm):
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
    def __init__(self, dataset_csv, datapath, chroma_type='hpcp', shortname='benchmark',
                oti=True, kappa=0.095, tau=1, m=9, downsample_fac=40):
        self.oti = oti
        self.tau = tau
        self.m = m
        self.chroma_type = chroma_type
        self.kappa = kappa
        self.tau = tau
        self.m = m
        self.downsample_fac = downsample_fac
        self.all_feats = {} # For caching features (global chroma and stacked chroma)
        CoverAlgorithm.__init__(self, dataset_csv=dataset_csv, name="Serra09", datapath=datapath, shortname=shortname)

    def load_features(self, i):
        if not i in self.all_feats:
            # Load features from file
            feats = CoverAlgorithm.load_features(self, i)
            # Load selected chroma feature
            chroma = feats[self.chroma_type]
            # Now downsample the chromas using median aggregation
            chroma = sync(chroma.T, np.arange(0, chroma.shape[0], self.downsample_fac), aggregate=np.median)
            self.all_feats[i] = chroma.T
        return self.all_feats[i]

    def similarity(self, idxs):
        for i,j in zip(idxs[:, 0], idxs[:, 1]):
            query = self.load_features(i)
            reference = self.load_features(j)
            # create instance of cover similarity algorithms from essentia
            crp_algo = ChromaCrossSimilarity(frameStackSize=self.m, 
                                            frameStackStride=self.tau, 
                                            binarizePercentile=self.kappa, 
                                            oti=self.oti)
            alignment_algo = CoverSongSimilarity(alignmentType='serra09', distanceType='symmetric')
            # compute similarity
            csm = crp_algo(query, reference)
            _, score = alignment_algo(csm)
            for key in self.Ds.keys():
                self.Ds[key][i][j] = score
    
    def normalize_by_length(self):
        """
        Do a non-symmetric normalization by length
        """
        for key in self.Ds.keys():
            for j in range(self.Ds[key].shape[1]):
                feature = self.load_features(j)
                norm_fac = np.sqrt(feature.shape[0])
                for i in range(self.Ds[key].shape[0]):     
                    # Do the reciprocal of what's written in the paper, since
                    # the evaluation statistics assume something with a higher
                    # score is more similar
                    self.Ds[key][i, j] /= norm_fac


def parser_args(args):
    parser = argparse.ArgumentParser(sys.argv[0], description="Benchmarking with Joan Serra's Cover id algorithm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '--dataset_csv', type=str, action="store",
                        help="Input dataset csv file")
    parser.add_argument("-d", '--datapath', type=str, action="store",
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="covers80",
                        help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default="hpcp",
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")

    return parser.parse_args(args)


if __name__ == '__main__':

    cmd_args = parser_args(sys.argv[1:])
    serra09 = Serra09(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname)
    serra09.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
    serra09.normalize_by_length()
    for similarity_type in serra09.Ds.keys():
        print(similarity_type)
        serra09.getEvalStatistics(similarity_type)
    serra09.cleanup_memmap()
    print("... Done ....")

