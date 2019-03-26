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
        # self.D = np.zeros((N, N))
        self.D = np.memmap('d_mat', shape=(N, N), mode='w+', dtype='float32')
        print("Initialized %s algorithm on %i songs"%(name, N))
    
    def load_features(self, i):
        """
        Load the fields from the h5 file for a particular
        song, and also keep track of which cover clique
        it's in by saving into self.cliques as a side effect
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
    
    def get_all_clique_ids(self, verbose=False):
        """
        Load all h5 files to get clique information as a side effect
        """
        import os
        if not os.path.exists("clique_info.txt"):
            fout = open("clique_info.txt", "w")
            for i in range(len(self.filepaths)):
                feats = CoverAlgorithm.load_features(self, i)
                if verbose:
                    print(i)
                print(feats['label'])
                fout.write("%i,%s\n"%(i, feats['label']))
            fout.close()
        else:
            fin = open("clique_info.txt")
            for line in fin.readlines():
                i, label = line.split(",")
                label = label.strip()
                if not label in self.cliques:
                    self.cliques[label] = set([])
                self.cliques[label].add(int(i))


    def similarity(self, i, j):
        """
        Given the indices of two songs, return a number
        which is high if the songs are similar, and low
        otherwise.
        Also store this number in D[i, j]
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
        return score
    
    
    def getEvalStatistics(self, topsidx = [1, 10, 100, 1000]):
        """
        Compute MR, MRR, MAP, Median Rank, and Top X
        """
        from itertools import chain
        D = np.array(self.D)
        N = D.shape[0]
        ## Step 1: Re-sort indices of D so that
        ## cover cliques are contiguous
        cliques = [list(self.cliques[s]) for s in self.cliques]
        Ks = np.array([len(c) for c in cliques]) # Length of each clique
        # Sort cliques in descending order of number
        idx = np.argsort(-Ks)
        Ks = Ks[idx]
        cliques = [cliques[i] for i in idx]
        # Unroll array of cliques and put distance matrix in
        # contiguous order
        idx = np.array(list(chain(*cliques)), dtype=int)
        D = D[idx, :]
        D = D[:, idx]
        
        ## Step 2: Compute MR, MRR, MAP, and Median Rank
        #Fill diagonal with -infinity to exclude song from comparison with self
        np.fill_diagonal(D, -np.inf)
        idx = np.argsort(-D, 1) #Sort row by row in descending order of score
        ranks = np.nan*np.ones(N)
        startidx = 0
        kidx = 0
        AllMap = np.nan*np.ones(N)
        for i in range(N):
            if i >= startidx + Ks[kidx]:
                startidx += Ks[kidx]
                kidx += 1
                print(startidx)
                if Ks[kidx] < 2:
                    # We're done once we get to a clique with less than 2
                    # since cliques are sorted in descending order
                    break
            iranks = []
            for k in range(N):
                diff = idx[i, k] - startidx
                if diff >= 0 and diff < Ks[kidx]:
                    iranks.append(k+1)
            iranks = iranks[0:-1] #Exclude the song itself, which comes last
            #For MR, MRR, and MDR, use first song in clique
            ranks[i] = iranks[0] 
            #For MAP, use all ranks
            P = np.array([float(j)/float(r) for (j, r) in \
                            zip(range(1, Ks[kidx]), iranks)])
            AllMap[i] = np.mean(P)
        MAP = np.nanmean(AllMap)
        ranks = ranks[np.isnan(ranks) == 0]
        print(ranks)
        MR = np.mean(ranks)
        MRR = 1.0/N*(np.sum(1.0/ranks))
        MDR = np.median(ranks)
        print("%s STATS\n-------------------------\nMR = %.3g\nMRR = %.3g\nMDR = %.3g\nMAP = %.3g"%(self.name, MR, MRR, MDR, MAP))
        tops = np.zeros(len(topsidx))
        for i in range(len(tops)):
            tops[i] = np.sum(ranks <= topsidx[i])
            print("Top-%i: %i"%(topsidx[i], tops[i]))
        return (MR, MRR, MDR, MAP, tops)

