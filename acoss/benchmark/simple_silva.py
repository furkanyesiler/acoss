# -*- coding: utf-8 -*-
"""
@2019
"""
import argparse
import scipy
import numpy as np
from librosa import util
from librosa import filters
from .algorithm_template import CoverAlgorithm


class Simple(CoverAlgorithm):
    """
    Attributes
    ----------
    Same as CoverAlgorithms, plus
    SSLEN=10, the length of subsequence used by SiMPle
    WIN=200, the window length for the dimensionality reduction
    SKIP=100, how many frames the dim reduction will skip each step
    """
    def __init__(self, dataset_csv, datapath="../features_covers80", chroma_type='hpcp', shortname='Covers80',
                 SSLEN=10, WIN=200, SKIP=100):
        self.SSLEN = SSLEN
        self.WIN = WIN
        self.SKIP = SKIP
        self.chroma_type = chroma_type
        CoverAlgorithm.__init__(self, dataset_csv=dataset_csv, name="Simple", datapath=datapath, shortname=shortname)

    def load_features(self, i, do_plot=False):
        feats = CoverAlgorithm.load_features(self, i)
        feat_orig = feats[self.chroma_type].T
        
        new_feat = np.zeros((feat_orig.shape[0], (int(feat_orig.shape[1]/self.SKIP))))

        for i in range(0, new_feat.shape[1]):
            new_feat[:, i] = np.mean(feat_orig[:, i*self.SKIP:i*self.SKIP+self.WIN], axis=1)

        return self.smooth(new_feat)
        
    def oti(self, seq_a, seq_b):
        profile_a = np.sum(seq_a, 1)
        profile_b = np.sum(seq_b, 1)
        oti_vec = np.zeros(12)
        for i in range(12):
            oti_vec[i] = np.dot(profile_a,np.roll(profile_b,i,axis=0))

        sorted_index = np.argsort(oti_vec)
        
        return np.roll(seq_b, sorted_index[-1], axis=0), sorted_index

    def smooth(self, feat, win_len_smooth = 4):
        '''
        This code is similar to the one used on librosa for smoothing cens: 
        https://librosa.github.io/librosa/generated/librosa.feature.chroma_cens.html
        '''
        win = filters.get_window('hann', win_len_smooth + 2, fftbins=False)
        win /= np.sum(win)
        win = np.atleast_2d(win)

        feat = scipy.signal.convolve2d(feat, win, mode='same', boundary='fill')
        return util.normalize(feat, norm=2, axis=0)
    
    def simple_sim(self, seq_a, seq_b):
    
        # prerequisites
        ndim = seq_b.shape[0]
        seq_a_len = seq_a.shape[1]
        seq_b_len = seq_b.shape[1]
        
        matrix_profile_len = seq_a_len - self.SSLEN + 1;
        
        # the "inverted" dot products will be used as the first value for reusing the dot products
        prods_inv = np.full([ndim,seq_a_len+self.SSLEN-1], np.inf)
        first_subseq = np.flip(seq_b[:,0:self.SSLEN],1)
            
        for i_dim in range(0,ndim):
            prods_inv[i_dim,:] = np.convolve(first_subseq[i_dim,:],seq_a[i_dim,:])
        prods_inv = prods_inv[:, self.SSLEN-1:seq_a_len]
           
        # windowed cumulative sum of the sequence b
        seq_b_cum_sum2 = np.insert(np.sum(np.cumsum(np.square(seq_b),1),0), 0, 0)
        seq_b_cum_sum2 = seq_b_cum_sum2[self.SSLEN:]-seq_b_cum_sum2[0:seq_b_len - self.SSLEN + 1]
        
        subseq_cum_sum2 = np.sum(np.square(seq_a[:,0:self.SSLEN]))
        
        # first distance profile
        first_subseq = np.flip(seq_a[:,0:self.SSLEN],1)
        dist_profile = seq_b_cum_sum2 + subseq_cum_sum2
        
        prods = np.full([ndim,seq_b_len+self.SSLEN-1], np.inf)
        for i_dim in range(0,ndim):
            prods[i_dim,:] = np.convolve(first_subseq[i_dim,:],seq_b[i_dim,:])
            dist_profile -= (2 * prods[i_dim,self.SSLEN-1:seq_b_len])
        prods = prods[:, self.SSLEN-1:seq_b_len] # only the interesting products
            
        matrix_profile = np.full(matrix_profile_len, np.inf)
        matrix_profile[0] = np.min(dist_profile)

        # for all the other values of the profile
        for i_subseq in range(1,matrix_profile_len):
            
            sub_value = seq_a[:,i_subseq-1, np.newaxis] * seq_b[:,0:prods.shape[1]-1]
            add_value = seq_a[:,i_subseq+self.SSLEN-1, np.newaxis] * seq_b[:, self.SSLEN:self.SSLEN+prods.shape[1]-1]
            
            prods[:,1:] = prods[:,0:prods.shape[1]-1] - sub_value + add_value
            prods[:,0] = prods_inv[:,i_subseq]
            
            subseq_cum_sum2 += -np.sum(np.square(seq_a[:,i_subseq-1])) + np.sum(np.square(seq_a[:,i_subseq+self.SSLEN-1]))
            dist_profile = seq_b_cum_sum2 + subseq_cum_sum2 - 2 * np.sum(prods,0)
            
            matrix_profile[i_subseq] = np.min(dist_profile)
        
        return np.median(matrix_profile)

    def similarity(self, idxs):
    
        for i,j in zip(idxs[:, 0], idxs[:, 1]):
            Si = self.load_features(i)
            Sj,_ = self.oti(Si,self.load_features(j))
            sim = -self.simple_sim(Si, Sj)
            self.Ds['main'][i, j] = sim


if __name__ == '__main__':
    # simple(chroma_type='crema')
    parser = argparse.ArgumentParser(description="Benchmarking with Similarity Matrix Profile-based similarity",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '--dataset_csv', type=str, action="store",
                        help="Input dataset csv file")
    parser.add_argument("-d", '--datapath', type=str, action="store", default='features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="Covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='crema',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")

    cmd_args = parser.parse_args()

    simple = Simple(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname)
    for i in range(len(simple.filepaths)):
        simple.load_features(i)
    print('Feature loading done.')
    simple.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=False)
    for similarity_type in simple.Ds.keys():
        simple.getEvalStatistics(similarity_type)
    simple.cleanup_memmap()

    print("... Done ....")
