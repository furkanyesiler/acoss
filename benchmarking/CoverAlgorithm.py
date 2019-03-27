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
    def __init__(self, name = "Generic", datapath="features_benchmark", shortname="full"):
        """
        Parameters
        ----------
        name: string
            Name of the algorithm
        datapath: string
            Path to folder with h5 files for the benchmark dataset
        shortname: string
            Short name for the dataset (for printing and saving results)
        """
        self.name = name
        self.shortname = shortname
        self.filepaths = glob.glob("%s/*.h5"%datapath)
        self.cliques = {}
        N = len(self.filepaths)
        # self.D = np.zeros((N, N))
        self.D = np.memmap('d_mat', shape=(N, N), mode='w+', dtype='float32')
        print("Initialized %s algorithm on %i songs in dataset %s"%(name, N, shortname))
    
    def load_features(self, i):
        """
        Load the fields from the h5 file for a particular
        song, and also keep track of which cover clique
        it's in by saving into self.cliques as a side effect
        NOTE: This function can be used to cache information
        about a particular song if that makes comparisons
        faster downstream (e.g. for FTM2D, cache the Fourier
        magnitude shingle median).  But this will not help
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
    
    def all_pairwise(self, parallel=0, n_cores=12, symmetric=False, precomputed=False):
        """
        Do all pairwise comparisons between songs, with code that is 
        amenable to parallelizations.
        In the serial case where features are cached, many algorithms will go
        slowly at the beginning but then speed up once the features for all
        songs have been computed
        Parameters
        ----------
        parallel: int
            If 0, run serial.  If 1, run parallel
        n_cores: int
            Number of cores to use in a parallel scenario
        symmetric: boolean
            Whether comparisons between pairs of songs are symmetric.  If so, the
            computation can be halved
        precomputed: boolean
            Whether all pairs have already been precomputed, in which case we just
            want to print the result statistics
        """
        from itertools import combinations, permutations
        import scipy.io as sio
        matfilename = "%s_%s.mat"%(self.name, self.shortname)
        if precomputed:
            D = sio.loadmat(matfilename)["D"]
            self.D = D
            self.get_all_clique_ids()
        else:
            pairs = range(len(self.filepaths))
            if symmetric:
                pairs = combinations(pairs, 2)
            else:
                pairs = permutations(pairs, 2)
            if parallel == 1:
                from joblib import Parallel, delayed
                Parallel(n_jobs=n_cores, verbose=1)(
                    delayed(self.similarity)(i, j) for idx, (i, j) in enumerate(pairs))
                self.get_all_clique_ids() # Since nothing has been cached
            else:
                for idx, (i, j) in enumerate(pairs):
                    self.similarity(i, j)
                    if idx%100 == 0:
                        print((i, j))
            if symmetric:
                self.D += self.D.T
            sio.savemat(matfilename, {"D":self.D})
        self.getEvalStatistics()
        if parallel == 1:
            import shutil
            try:
                shutil.rmtree('d_mat')
            except:  # noqa
                print('Could not clean-up automatically.')
    
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
