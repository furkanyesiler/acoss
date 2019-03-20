"""
A template class for all benchmarking algorithms
"""

import numpy as np
import glob
import deepdish as dd

class CoverAlgorithm(object):
    """
    Attributes
    ----------
    filepaths: list(string)
        List of paths to all files in the dataset
    cliques: {string: set}
        A dictionary of all cover cliques, where the cliques
        index into filepaths
    D: ndarray(num files, num files)
        A pairwise similarity matrix, whose indices index
        into filepaths.  Assumed to be symmetric
    """
    def __init__(self, name = "Generic", datapath="features_benchmark"):
        """
        Parameters
        ----------
        datapath: string
            Path to folder with h5 files for the benchmark dataset
        """
        self.name = name
        self.filepaths = glob.glob("%s/*.h5"%datapath)
        self.cliques = {}
        N = len(self.filepaths)
        self.D = np.zeros((N, N))
        print("Initialized %s algorithm on %i songs"%(name, N))
    
    def load_features(self, i):
        """
        Load the fields from the h5 file for a particular
        song, and also keep track of which cover clique
        it's in.  
        NOTE: This function can be used to cache information
        about a particular song if that makes comparisons
        faster downstream (e.g. for FTM2D, cache the Fourier
        magnitude shingle median).  But this may not help as much
        in a parallel scenario
        Parameters
        ----------
        i: int
            Index of song in self.filepaths
        Returns
        -------
        feats: dictionary
            Dictionary of features for the song
        """
        feats = dd.io.load(self.filepaths[i])
        # Keep track of what cover clique it's in
        if not feats['label'] in self.cliques:
            self.cliques[feats['label']] = set([])
        self.cliques[feats['label']].add(i)
        return feats


    def similarity(self, i, j):
        """
        Given the indices of two songs, return a number
        which is high if the songs are similar, and low
        otherwise.  It is assumed that this score is symmetric,
        so it only needs to be computed for i, j: j > i
        Also store this number in D[i, j] and D[j, i] 
        as a side effect
        Parameters
        ----------
        i: int
            Index of first song in self.filepaths
        j: int
            Index of second song in self.filepaths
        """
        score = 0.0
        self.D[i, j] = score
        self.D[j, i] = score
        return score
    
    def getEvalStatistics(self, topsidx = [10, 100, 1000]):
        """
        Compute MR, MRR, MAP, Median Rank, and Top X
        """
        from itertools import chain
        D = np.array(self.D)
        N = D.shape[0]
        ## Step 1: Re-sort indices of D so that
        ## cover cliques are contiguous
        cliques = [list(self.cliques[s]) for s in self.cliques]
        Ks = [len(c) for c in cliques] # Length of each clique
        idx = np.array(list(chain(*cliques)), dtype=int)
        D = D[idx, :]
        D = D[:, idx]
        
        ## Step 2: Compute MR, MRR, MAP, and Median Rank
        #Fill diagonal with -infinity to exclude song from comparison with self
        np.fill_diagonal(D, -np.inf)
        idx = np.argsort(-D, 1) #Sort row by row in descending order of score
        ranks = np.zeros(N)
        startidx = 0
        kidx = 0
        for i in range(N):
            if i >= startidx + Ks[kidx]:
                startidx += Ks[kidx]
                kidx += 1
            print(startidx)
            for k in range(N):
                diff = idx[i, k] - startidx
                if diff >= 0 and diff < Ks[kidx]:
                    ranks[i] = k+1
                    break
        print(ranks)
        MR = np.mean(ranks)
        MRR = 1.0/N*(np.sum(1.0/ranks))
        MDR = np.median(ranks)
        print("%s STATS\n-------------------------\nMR = %g\nMRR = %g\nMDR = %g\n"%(self.name, MR, MRR, MDR))
        tops = np.zeros(len(topsidx))
        for i in range(len(tops)):
            tops[i] = np.sum(ranks <= topsidx[i])
            print("Top-%i: %i"%(topsidx[i], tops[i]))
        return (MR, MRR, MDR, tops)
