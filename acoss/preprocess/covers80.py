"""
Compute features for the covers80 dataset.  This assumes
that the mp3 files have been downloaded from 
https://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz
and the subdirectory "covers32" has been placed at the root of this directory
"""
from preprocess.extractors import compute_features_from_list_file


def compute_covers80_features():
    fin = open("covers32k/list1.list")
    files = fin.readlines()
    fin.close()
    fin = open("covers32k/list2.list")
    files += fin.readlines()
    fin.close()
    files = ["covers32k/%s.mp3"%s.strip() for s in files]
    fout = open("covers80.txt", "w")
    for f in files:
        fout.write("%s\n"%f)
    fout.close()
    compute_features_from_list_file("covers80.txt", "covers80/")


if __name__ == '__main__':
    compute_covers80_features()