import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import pandas as pd
import seaborn as sns
import glob

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
    for i, f in enumerate(files):
        if i%100 == 0:
            print("Loaded %i of %i..."%(i, len(files)))
        fields = dd.io.load(f)
        label, feat = fields['label'], extractor(fields)
        if not label in pairs:
            pairs[label] = []
        pairs[label].append(feat)
    return pairs

def save_keys_csv():
    """
    Save a csv file called "keys.csv" with the extracted keys
    for all pairs
    """
    pairs = get_cover_pairs(lambda x: x['key_extractor'])
    table = []
    index = []
    for p in pairs:
        index.append(p)
        s1 = pairs[p][0]
        s2 = pairs[p][1]
        table.append([s1['key'], s1['scale'], s1['strength'], s2['key'], s2['scale'], s2['strength']])
    df = pd.DataFrame(table, index=index, columns=['Key1', 'Scale1', 'Strength1', 'Key2', 'Scale2', 'Strength2'])
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

    # For those that change key, what is the distribution
    # of the absolute transposition distance?
    key2idx = {'C':0, 'C#':1, 'D':2, 'Eb':3, 'E':4, 'F':5, 'F#':6, 'G':7, 'Ab':8, 'A':9, 'Bb':10, 'B':11}
    keyidx = []
    for i in range(key.shape[0]):
        keyidx.append([key2idx[c] for c in key[i, :]])
    keyidx = np.array(keyidx)
    keyidx = keyidx[keysame == 0, :]
    dist = np.abs(keyidx[:, 0] - keyidx[:, 1])
    dist = np.minimum(dist, 12-dist)
    plt.clf()
    sns.distplot(dist, kde=False, norm_hist=False)
    plt.xlabel("Transposition Distance")
    plt.ylabel("Count")
    plt.title("Distribution of Transposition Changes")
    plt.savefig("Transposition.svg", bbox_inches='tight')



if __name__ == '__main__':
    #save_keys_csv()
    get_key_stats()