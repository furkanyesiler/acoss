# -*- coding: utf-8 -*-
"""
"""
from sklearn.metrics.pairwise import euclidean_distances
from pySeqAlign import qmax
from CoverAlgorithm import *
import numpy as np
import argparse


def global_hpcp(chroma):
    """Computes global hpcp of a input chroma vector"""
    if chroma.shape[1] not in [12, 24, 36]:
        raise IOError("Wrong axis for the input chroma array. Expected shape '(frame_size, bin_size)'")
    return np.divide(chroma.sum(axis=0), np.max(chroma.sum(axis=0)))


def optimal_transposition_index(chromaA, chromaB, n_shifts=12):
    """
    Computes optimal transposition index (OTI) for the chromaB to be transposed in the same key as of chromaA
    Input :
            chromaA : chroma feature array of the query song for reference key
            chromaB : chroma feature array of the reference song for which OTI has to be applied
        Params:
                n_shifts: (default=12) Number of oti tranpositions to be checked for circular shift
    Output : Integer value specifying optimal transposition index for transposing chromaB to chromaA to be in same key
    """
    global_hpcpA = global_hpcp(chromaA)
    global_hpcpB = global_hpcp(chromaB)
    idx = list()
    for index in range(n_shifts):
        idx.append(np.dot(global_hpcpA, np.roll(global_hpcpB, index)))
    return int(np.argmax(idx))


def transpose_by_oti(chromaB, oti=0):
    """
    Transpose the chromaB vector to a common key by a value of optimal transposition index
    Input :
            chromaB : input chroma array        
    Output : chromaB vector transposed to a factor of specified OTI
    """
    return np.roll(chromaB, oti)


def to_embedding(input_array, tau=1, m=9):
    """
    Construct a time series with delay embedding 'tau' and embedding dimension 'm' from the input audio feature vector
    Input :
            input_array : input feature array for constructing the stacked features
    Params :
            tau (default : 1): delay embedding
            m (default : 9): embedding dimension

    Output : Time series representation of the input audio feature vector
    """

    timeseries = list()
    for startTime in range(0, input_array.shape[0] - m*tau, tau):
        stack = list()
        for idx in range(startTime, startTime + m*tau, tau):
            stack.append(input_array[idx])
        timeseries.append(np.ndarray.flatten(np.array(stack)))
    return np.array(timeseries)



def cross_recurrent_plot(input_x, input_y, tau=1, m=9, kappa=0.095, oti=True):
    """
    Constructs the Cross Recurrent Plot of two audio feature vector as mentioned in [1]
    Inputs :
            input_x : input feature array of query song
            input_y : input feature array of reference song
    Params :
            kappa (default=0.095)       : fraction of mutual nearest neighbours to consider [0, 1]
            tau (default=1)             : delay embedding [1, inf]
            m (default=9)               : embedding dimension for the time series embedding [0, inf]
            swapaxis (default=False)    : swapaxis of the feature array if it not in the shape (x,12) where x is the \
                                          time axis
            oti (default=True)    : boolean to check to choose if OTI should be applied to the reference song

    Output : Binary similarity matrix where 1 constitutes a similarity between
            two feature vectors at ith and jth position respectively and 0 denotes non-similarity
    """

    if oti:
        oti_idx = optimal_transposition_index(input_x, input_y)
        input_y = transpose_by_oti(input_y, oti_idx) #transpose input_y to the key of input_x by a oti value

    pdistances = euclidean_distances(input_x, input_y)
    transposed_pdistances = pdistances.T

    eph_x = np.percentile(pdistances, kappa*100, axis=1)
    eph_y = np.percentile(transposed_pdistances, kappa*100, axis=1)
    x = eph_x[:,None] - pdistances
    y = eph_y[:,None] - transposed_pdistances

    #apply step function to the array (Binarize the array)
    x = np.piecewise(x, [x<0, x>=0], [0,1])
    y = np.piecewise(y, [y<0, y>=0], [0,1])

    crp = x*y.T
    return crp



class Serra09(CoverAlgorithm):
    """
    Attributes
    ----------
    Same as CoverAlgorithms, plus
    chroma_type: string
        Type of chroma to use (key into features)
    shapes: {int: int}
        Shapes of each song, used for normalization
    """
    def __init__(self, datapath="../features_covers80", chroma_type='hpcp', shortname='benchmark', 
                oti=True, kappa=0.095, tau=1, m=9):

        self.oti = oti
        self.tau = tau
        self.m = m
        self.chroma_type = chroma_type
        self.kappa = kappa
        self.tau = tau
        self.m = m
        self.shapes = {}
        CoverAlgorithm.__init__(self, "QMAX", datapath=datapath, shortname=shortname)

    def load_features(self, i):
        feats = CoverAlgorithm.load_features(self, i)
        chroma = feats[self.chroma_type].T
        return to_embedding(chroma, tau=self.tau, m=self.m)

    def similarity(self, idxs):
        print(idxs)
        for i,j in zip(idxs[:, 0], idxs[:, 1]):
            query_feature = self.load_features(i)
            reference_feature = self.load_features(j)
            csm = cross_recurrent_plot(query_feature, reference_feature, kappa=self.kappa, oti=self.oti)
            csm = csm.astype(dtype=np.uint8)
            M, N = csm.shape[0], csm.shape[1]
            D = np.zeros(M*N, dtype=np.float32)
            score = qmax(csm.flatten(), D, M, N)
            self.shapes[j] = N
            #score = np.sqrt(csm.shape[1]) / score
            for key in self.Ds.keys():
                self.Ds[key][i][j] = score
    
    def normalize_by_length(self):
        """
        Do a non-symmetric normalization by length
        """
        for key in self.Ds.keys():
            for i in range(self.Ds[key].shape[0]):
                for j in range(self.Ds[key].shape[1]):
                    self.Ds[key][i, j] = np.sqrt(self.shapes[j]) / self.Ds[key][i, j]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with Joan Serra's Cover id algorithm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")

    cmd_args = parser.parse_args()

    qmax = Serra09(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname)
    qmax.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
    qmax.normalize_by_length()
    for similarity_type in qmax.Ds.keys():
        print(similarity_type)
        qmax.getEvalStatistics(similarity_type)
    qmax.cleanup_memmap()
    print("... Done ....")

