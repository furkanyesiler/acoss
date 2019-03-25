"""
A template class for all benchmarking algorithms
"""

import numpy as np
import glob
import deepdish as dd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

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

    def MAP(self):
        """
           Compute MAP
        """
        # number of samples in the dataset
        num_of_samples = len(self.filepaths)

        labels = np.empty(num_of_samples, dtype='U32')

        dist_matrix = np.array(self.D)
        dist_matrix = np.array(-dist_matrix)
        np.fill_diagonal(dist_matrix, np.inf)

        for s in self.cliques:
            temp_list = list(self.cliques[s])
            for i in range(len(temp_list)):
                labels[temp_list[i]] = s

        tuple_dtype = np.dtype([('f1', np.float), ('f2', np.unicode_, 32)])

        # initializing a matrix to store tuples of pairwise distances and labels of the reference samples
        tuple_matrix = np.ndarray(shape=(num_of_samples, num_of_samples), dtype=tuple_dtype)

        # filling the tuple_matrix with distance values and labels
        for i in range(num_of_samples):
            for j in range(num_of_samples):
                tuple_matrix[i][j] = (dist_matrix[i][j], labels[j])

        # initializing mAP
        mAP = 0

        # calculating average precision for each row of the distance matrix
        for i in range(num_of_samples):
            # obtaining the current row
            row = tuple_matrix[i]

            # label of the current query
            label = labels[i]

            # sorting the row with respect to distance values
            row.sort(order='f1')

            # initializing true positive count
            tp = 0

            # initializing precision value
            prec = 0

            # counting number of instances that has the same label as the query
            label_count = 0

            for j in range(1, num_of_samples):
                # checking whether the reference sample has the same label as the query
                temp_var = row[j][1]
                if row[j][1] == label:

                    # incrementing the number of true positives
                    tp += 1

                    # updating the precision value
                    prec += tp / j

                    # incrementing the number of samples with the same label as the query
                    label_count += 1

            # updating  mAP
            mAP += prec / label_count

        # updating mAP
        mAP = mAP / num_of_samples

        return mAP
    
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
            #print(startidx)
            for k in range(N):
                diff = idx[i, k] - startidx
                if diff >= 0 and diff < Ks[kidx]:
                    ranks[i] = k+1
                    break
        #print(ranks)
        MR = np.mean(ranks)
        MRR = 1.0/N*(np.sum(1.0/ranks))
        MDR = np.median(ranks)
        print("%s STATS\n-------------------------\nMR = %g\nMRR = %g\nMDR = %g\nMAP = %.2g\n"%(self.name, MR, MRR, MDR, self.MAP()))
        tops = np.zeros(len(topsidx))
        for i in range(len(tops)):
            tops[i] = np.sum(ranks <= topsidx[i])
            print("Top-%i: %i"%(topsidx[i], tops[i]))
        return (MR, MRR, MDR, tops)
