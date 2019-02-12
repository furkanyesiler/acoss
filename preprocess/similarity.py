# -*- coding: utf-8 -*-
"""
Here we can different methods to compute similarity matrix and distances

TODO: add methods from Chris and Diego 
"""
from sklearn.metrics.pairwise import euclidean_distances
from essentia.standard import CoverSongSimilarity
from essentia import array
import numpy as np


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


def to_embedding(input_xrray, tau=1, m=9):
    """
    Construct a time series with delay embedding 'tau' and embedding dimension 'm' from the input audio feature vector
    Input :
            input_xrray : input feature     array for constructing the timespace embedding
    Params :
            tau (default : 1): delay embedding
            m (default : 9): embedding dimension

    Output : Time series representation of the input audio feature vector
    """

    timeseries = list()
    for startTime in range(0, input_xrray.shape[0] - m*tau, tau):
        stack = list()
        for idx in range(startTime, startTime + m*tau, tau):
            stack.append(input_xrray[idx])
        timeseries.append(np.ndarray.flatten(np.array(stack)))
    return np.array(timeseries)


def cross_recurrent_plot(input_x, input_y, tau=1, m=9, kappa=0.095, transpose=True):
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
            transpose (default=True)    : boolean to check to choose if OTI should be applied to the reference song

    Output : Binary similarity matrix where 1 constitutes a similarity between
            two feature vectors at ith and jth position respectively and 0 denotes non-similarity
    """

    if transpose:
        oti_idx = optimal_transposition_index(input_x, input_y)
        input_y = transpose_by_oti(input_y, oti_idx) #transpose input_y to the key of input_x by a oti value

    timespaceA = to_embedding(input_x, tau=tau, m=m)
    timespaceB = to_embedding(input_y, tau=tau, m=m)

    pdistances = euclidean_distances(timespaceA, timespaceB)
    #pdistances = resample_simmatrix(pdistances)
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


def serra_cover_similarity_measures(input_crp, disOnset=0.5, disExtension=0.5, simType='qmax'):
    """
    Computes distance cover song similarity measure using smith-waterman local allignment from the
    cross recurrent plots as mentioned in [1] (qmax) and [2] (dmax)

    [1]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover
        song identification. New Journal of Physics, 11.

    [2]. Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification.
         Multimedia Tools and Applications.

    Input:
        input_crp: 2-d binary matrix of cross recurrent plot (x-axis query song and y-axis for reference song)

      Params:
             disOnset: penalty for a disurption onset
             disExtension: penalty for a disurption extension
             simType: ['qmax', 'dmax']

    Return: cover similarity distance

    NOTE: CoverSongSimilarity algo will be available soon in the new essentia release
    """
    coversim = CoverSongSimilarity(disOnset=disOnset, disExtension=disExtension, simType=simType)
    score_matrix = coversim.compute(array(input_crp))
    return np.divide(np.sqrt(input_crp.shape[1]), np.max(score_matrix))

