"""
A template class for all benchmark algorithms

"""
import numpy as np
import glob
import os
import deepdish as dd
import warnings

from ..utils import create_dataset_filepaths


class CoverAlgorithm(object):
    """
    Attributes
    ----------
    filepaths: list(string)
        List of paths to all files in the dataset
    cliques: {string: set}
        A dictionary of all cover cliques, where the cliques
        index into filepaths
    Ds: {string silarity type: ndarray(num files, num files)}
        A dictionary of pairwise similarity matrices, whose 
        indices index into filepaths.
    """
    def __init__(self,
                 dataset_csv,
                 name="Generic",
                 datapath="features_benchmark",
                 shortname="full",
                 cachedir="cache",
                 similarity_types=["main"]):
        """
        Parameters
        ----------
        name: string
            Name of the algorithm
        datapath: string
            Path to folder with h5 files for the benchmark dataset
        shortname: string
            Short name for the dataset (for printing and saving results)
        cachedir: string
            Directory to which to cache intermediate feature computations, etc
        """
        self.name = name
        self.shortname = shortname
        self.cachedir = cachedir
        self.filepaths = create_dataset_filepaths(dataset_csv, root_audio_dir=datapath, file_format=".h5")
        self.cliques = {}
        self.N = len(self.filepaths)
        if not os.path.exists(cachedir):
            os.mkdir(cachedir)
        self.Ds = {}
        for s in similarity_types:
            self.Ds[s] = np.memmap('%s_%s_dmat' % (self.get_cacheprefix(), s), shape=(self.N, self.N), mode='w+', dtype='float32')
        print("Initialized %s algorithm on %i songs in dataset %s" % (name, self.N, shortname))
    
    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s" % (self.cachedir, self.name, self.shortname)

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
        filepath = "%s_clique_info.txt" % self.get_cacheprefix()
        if not os.path.exists(filepath):
            fout = open(filepath, "w")
            for i in range(len(self.filepaths)):
                feats = CoverAlgorithm.load_features(self, i)
                if verbose:
                    print(i)
                print(feats['label'])
                fout.write("%i,%s\n"%(i, feats['label']))
            fout.close()
        else:
            fin = open(filepath)
            for line in fin.readlines():
                i, label = line.split(",")
                label = label.strip()
                if not label in self.cliques:
                    self.cliques[label] = set([])
                self.cliques[label].add(int(i))

    def similarity(self, idxs):
        """
        Given the indices of two songs, return a number
        which is high if the songs are similar, and low
        otherwise, for each similarity type
        Also store this number in D[i, j]
        as a side effect
        Parameters
        ----------
        i: int
            Index of first song in self.filepaths
        j: int
            Index of second song in self.filepaths
        """
        (a, b) = idxs.shape
        for k in range(a):
            i = idxs[k][0]
            j = idxs[k][1]
            score = 0.0
            self.Ds["main"][i, j] = score
    
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
        h5filename = "%s_Ds.h5" % self.get_cacheprefix()
        if precomputed:
            self.Ds = dd.io.load(h5filename)
            self.get_all_clique_ids()
        else:
            if symmetric:
                all_pairs = [(i, j) for idx, (i, j) in enumerate(combinations(range(len(self.filepaths)), 2))]
            else:
                all_pairs = [(i, j) for idx, (i, j) in enumerate(permutations(range(len(self.filepaths)), 2))]
            chunks = np.array_split(all_pairs, 45)
            if parallel == 1:
                from joblib import Parallel, delayed
                Parallel(n_jobs=n_cores, verbose=1)(
                    delayed(self.similarity)(chunks[i]) for i in range(len(chunks)))
                self.get_all_clique_ids() # Since nothing has been cached
            else:
                for idx, (i, j) in enumerate(all_pairs):
                    self.similarity(np.array([[i, j]]))
                    if idx % 100 == 0:
                        print((i, j))
            if symmetric:
                for similarity_type in self.Ds:
                    self.Ds[similarity_type] += self.Ds[similarity_type].T
            dd.io.save(h5filename, self.Ds)    

    def cleanup_memmap(self):
        """
        Remove all memmap variables for song-level similarity matrices
        """
        import shutil
        try:
            for s in self.Ds:
                shutil.rmtree('%s_%s_dmat'%(self.get_cacheprefix(), s))
        except:
            print('Could not clean-up automatically.')
    
    def getEvalStatistics(self, similarity_type, topsidx=[1, 10, 100, 1000]):
        """
        Compute MR, MRR, MAP, Median Rank, and Top X using
        a particular similarity measure
        Parameters
        ----------
        similarity_type: string
            The similarity measure to use
        """
        from itertools import chain
        D = np.array(self.Ds[similarity_type], dtype=np.float32)
        N = D.shape[0]
        # Step 1: Re-sort indices of D so that
        # cover cliques are contiguous
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
        
        # Step 2: Compute MR, MRR, MAP, and Median Rank
        # Fill diagonal with -infinity to exclude song from comparison with self
        np.fill_diagonal(D, -np.inf)
        idx = np.argsort(-D, 1) # Sort row by row in descending order of score
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
            iranks = iranks[0:-1] # Exclude the song itself, which comes last
            if len(iranks) == 0:
                warnings.warn("Recalling 0 songs for clique of size %i at song index %i"%(Ks[kidx], i))
                break
            # For MR, MRR, and MDR, use first song in clique
            ranks[i] = iranks[0] 
            # For MAP, use all ranks
            P = np.array([float(j)/float(r) for (j, r) in \
                            zip(range(1, Ks[kidx]), iranks)])
            AllMap[i] = np.mean(P)
        MAP = np.nanmean(AllMap)
        ranks = ranks[np.isnan(ranks) == 0]
        print(ranks)
        MR = np.mean(ranks)
        MRR = 1.0/N*(np.sum(1.0/ranks))
        MDR = np.median(ranks)
        print("%s %s STATS\n-------------------------\nMR = %.3g\nMRR = %.3g\nMDR = %.3g\nMAP = %.3g"
              % (self.name, similarity_type, MR, MRR, MDR, MAP))
        tops = np.zeros(len(topsidx))
        for i in range(len(tops)):
            tops[i] = np.sum(ranks <= topsidx[i])
            print("Top-%i: %i"%(topsidx[i], tops[i]))
        
        # Output to CSV file
        resultsfile = "results_%s.csv"%self.shortname
        if not os.path.exists(resultsfile):
            fout = open(resultsfile, "w")
            fout.write("name, MR, MRR, MDR, MAP")
            for t in topsidx:
                fout.write(",Top-%i"%t)
            fout.write("\n")
        fout = open(resultsfile, "a")
        fout.write("%s_%s,"%(self.name, similarity_type))
        fout.write("%.3g, %.3g, %.3g, %.3g"%(MR, MRR, MDR, MAP))
        for t in tops:
            fout.write(", %.3g"%t)
        fout.write("\n")
        fout.close()
        return MR, MRR, MDR, MAP, tops

