# -*- coding: utf-8 -*-
"""
@2019
"""
import glob
import matplotlib.pyplot as plt
import deepdish as dd
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from ..benchmark.utils.cross_recurrence import *


def get_cover_pairs(extractor):
    """
    Retrieve all of a particular feature
    Parameters
    ----------
    extractor: function: h5 dictionary -> feature
        A function that extracts some feature
        from a dictionary of h5 fields
    Returns
    -------
    pairs: dictionary: ID -> [feature1, feature2]
        A dictionary from cover pair ID to a pair
        of features
    """
    files = glob.glob('feature_whatisacover/*.h5')
    pairs = {}
    paths = {}
    for i, f in enumerate(files):
        if i % 100 == 0:
            print("Loaded %i of %i..." % (i, len(files)))
        fields = dd.io.load(f)
        label, feat = fields['label'], extractor(fields)
        if not label in pairs:
            pairs[label] = []
            paths[label] = []
        pairs[label].append(feat)
        paths[label].append(f)
    return pairs, paths


def get_key_info(fields):
    ret = fields['key_extractor']
    ret['track_id'] = fields['track_id']
    return ret


def save_keys_csv():
    """
    Save a csv file called "keys.csv" with the extracted keys
    for all pairs
    """
    pairs, _ = get_cover_pairs(get_key_info)
    table = []
    index = []
    for p in pairs:
        index.append(p)
        s1 = pairs[p][0]
        s2 = pairs[p][1]
        table.append([s1['track_id'], s1['key'], s1['scale'], s1['strength'], s2['track_id'], s2['key'], s2['scale'], s2['strength']])
    df = pd.DataFrame(table, index=index, columns=['ID1', 'Key1', 'Scale1', 'Strength1', 'ID2', 'Key2', 'Scale2', 'Strength2'])
    df.to_csv("keys.csv")


def get_key_stats(min_confidence=0.75):
    """
    Look at the following:
    * Histogram of confidences
    * Proportion of songs that stay major or minor
    * Proportion of songs that stay in the same key
    * Distribution of transposition distance for songs that are in the same key
    """
    res = pd.read_csv('keys.csv')
    # Histogram of confidences
    x = res[['Strength1', 'Strength2']].values
    sns.distplot(np.min(x, 1), norm_hist=True)
    plt.xlabel("Strength")
    plt.ylabel("Density")
    plt.title("Histogram of Minimum Key Confidences for A Pair")
    plt.savefig("KeyConfidences.svg", bbox_inches='tight')
    idx = np.min(x, 1) > min_confidence
    print("%i pairs exceed minimum of %g confidence"%(np.sum(idx), min_confidence))
    
    # Look at major/minor
    majmin = res[['Scale1', 'Scale2']].values[idx == 1, :]
    majmin = majmin[:, 0] == majmin[:, 1]
    print("%.3g %% same major/minor"%(100*np.sum(majmin)/float(majmin.shape[0])))
    plt.clf()
    sns.countplot(data=pd.DataFrame(majmin, columns=['Key Changes']), x="Key Changes")
    plt.savefig("MajorMinor.svg", bbox_inches='tight')

    # Among those that stay major/minor, which ones stay in the same key?
    key = res[['Key1', 'Key2']].values[idx == 1, :]
    keysame = majmin*(key[:, 0] == key[:, 1])
    print("%.3g %% same key"%(100*np.sum(keysame)/float(keysame.shape[0])))
    plt.clf()
    sns.countplot(data=pd.DataFrame(keysame, columns=['Key Changes']), x="Key Changes")
    plt.savefig("KeySame.svg", bbox_inches='tight')

    # Pull out a list of songs that change keys
    songsidx = np.arange(majmin.shape[0])
    songsidx = songsidx[majmin == 0]
    songs = res[['ID1', 'ID2', 'Key1', 'Scale1', 'Key2', 'Scale2']].values[idx == 1, :]
    songs = songs[songsidx, :]
    songs = songs[np.random.permutation(songs.shape[0]), :]
    songs = songs[0:50, :]
    for i in range(songs.shape[0]):
        for k in [0, 1]:
            songs[i, k] = songs[i, k].replace("P_", "https://secondhandsongs.com/performance/")
    df = pd.DataFrame(songs, columns=['URL1', 'URL2', 'Key1', 'Scale1', 'Key2', 'Scale2'])
    df.to_csv("majorminor.csv")
    

    # For those that change key, what is the distribution
    # of the absolute transposition distance?
    key2idx = {'C':0, 'C#':1, 'D':2, 'Eb':3, 'E':4, 'F':5, 'F#':6, 'G':7, 'Ab':8, 'A':9, 'Bb':10, 'B':11}
    keyidx = []
    for i in range(key.shape[0]):
        keyidx.append([key2idx[c] for c in key[i, :]])
    keyidx = np.array(keyidx)
    songidxs = np.arange(majmin.shape[0])
    keyidx = keyidx[(keysame==0)*(majmin==1), :]
    dist = np.abs(keyidx[:, 0] - keyidx[:, 1])
    dist = np.minimum(dist, 12-dist)
    plt.figure(figsize=(2.5, 2.5))
    sns.distplot(dist, kde=False, norm_hist=False)
    plt.xlabel("Transposition Distance in Halfsteps")
    plt.ylabel("Count")
    plt.title("Transposition Changes")
    plt.savefig("Transposition.svg", bbox_inches='tight')


def get_maxtempo(row):
    x = row['madmom_features']['tempos']
    return x[np.argmax(x[:, 1]), :]


def save_tempo_csv():
    """
    Save a csv file called "tempos.csv" with the extracted clearest
    tempos and their confidences
    """
    pairs, _ = get_cover_pairs(get_maxtempo)
    table = np.zeros((len(pairs), 4))
    index = []
    for i, p in enumerate(pairs):
        index.append(p)
        table[i, 0:2] = pairs[p][0]
        table[i, 2::] = pairs[p][1]
    df = pd.DataFrame(table, index=index, columns=['Tempo1', 'Strength1', 'Tempo2', 'Strength2'])
    df.to_csv("tempos.csv")


def get_tempo_stats(min_confidence=0):
    """
    Look at the following:
    * Histogram of confidences
    * Distribution of tempo ratios
    """
    res = pd.read_csv('tempos.csv')
    # Histogram of confidences
    x = res[['Strength1', 'Strength2']].values
    sns.distplot(np.min(x, 1), norm_hist=True)
    plt.xlabel("Strength")
    plt.ylabel("Density")
    plt.title("Histogram of Minimum Tempo Confidences for A Pair")
    plt.savefig("TempoConfidences.svg", bbox_inches='tight')
    idx = np.min(x, 1) > min_confidence
    print("%i pairs exceed minimum of %g confidence"%(np.sum(idx), min_confidence))
    
    # Histogram of tempo ratios
    tempos = res[['Tempo1', 'Tempo2']].values
    ratios = tempos[:, 1]/tempos[:, 0]
    ratios[ratios < 1] = 1.0/ratios[ratios < 1]
    print(np.quantile(ratios, 0.25))
    print(np.quantile(ratios, 0.5))
    print(np.quantile(ratios, 0.75))
    plt.figure(figsize=(2.5, 2.5))
    sns.distplot(ratios, norm_hist=False)
    plt.xlim([1, 2.2])
    plt.xlabel("Ratio")
    plt.ylabel("Counts")
    plt.title("Tempo Ratios")
    plt.savefig("TempoRatios.svg", bbox_inches='tight')


def getFMeasure(tags1, tags2, cutoff = 0.062):
    tags1 = {s:f for (s, f) in tags1 if float(f) > cutoff} 
    tags2 = {s:f for (s, f) in tags2 if float(f) > cutoff}
    if len(tags1) == 0 or len(tags2) == 0:
        return np.inf
    r1 = 0
    for t in tags1:
        if t in tags2:
            r1 += 1
    r2 = 0
    for t in tags2:
        if t in tags1:
            r2 += 1
    r = float(r1)/len(tags1)
    p = float(r2)/len(tags2)
    if r == 0 or p == 0:
        return 0
    return 2*(r*p)/(r+p)
    

def get_tag_stats():
    """
    Look at the F-measure between tags of cover pairs and false
    cover pairs
    """
    import scipy.io as sio
    files = glob.glob('tag_all_whatisacover/*.h5')
    pairs = {}
    for i, f in enumerate(files):
        if i%100 == 0:
            print("Loaded %i of %i..."%(i, len(files)))
        fields = dd.io.load(f)
        label, tags = fields['label'], fields['tags']
        if not label in pairs:
            pairs[label] = []
        pairs[label].append(tags)
    cutoff = 0.062
    true_pairs = np.zeros(len(pairs))
    false_pairs = []
    keys = list(pairs.keys())
    for i, k in enumerate(keys):
        print(i)
        tags1 = pairs[k][0]
        tags2 = pairs[k][1]
        true_pairs[i] = getFMeasure(tags1, tags2, cutoff)
        for k2 in keys:
            if k == k2:
                continue
            false_pairs.append(getFMeasure(tags1, pairs[k2][1], cutoff))
    false_pairs = np.array(false_pairs)
    true_pairs = true_pairs[np.isfinite(true_pairs)]
    false_pairs = false_pairs[np.isfinite(false_pairs)]
    sio.savemat("tags.mat", {"true_pairs":true_pairs, "false_pairs":false_pairs})
    plt.figure(figsize=(8, 4))
    sns.distplot(true_pairs, kde=False, bins=bins, norm_hist=True)
    sns.distplot(false_pairs, kde=False, bins=bins, norm_hist=True)
    plt.xlabel("F-Measure")
    plt.ylabel("Density")
    plt.legend(["True Pairs", "False Pairs"])
    plt.title("Auto Tagging F-Measure Distributions")
    plt.show()
    plt.savefig("AutoTag.svg", bbox_inches='tight')
    print(ks_2samp(true_pairs, false_pairs))


if __name__ == '__main__':
    # save_keys_csv()
    get_key_stats()
    # save_tempo_csv()
    get_tempo_stats()
    # get_onset_stats()
    # get_tag_stats()
