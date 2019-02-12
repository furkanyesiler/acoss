# -*- coding: utf-8 -*-
"""
[TODO]: Here we can add batch feature extractors for all audio files in the dataset with multiple threads

The following functions are some examples and we can refactor it to meet our needs (add more features or store in more optimised fileformats etc)

"""
from joblib import Parallel, delayed
from .utils import timeit, log, read_txt_file
from .features import AudioFeatures
import argparse
import time
import glob
import json
import os


_LOG_FILE = log("<path_to_log_file>")


PROFILE = {
           'sample_rate': 44100,
           'downsample_audio': False,
           'downsample_factor': 2,
           'features': ['cqt_nsg', 'chroma_cqt', 'chroma_cens', 'hpcp', 'tempogram', 'two_d_fft_mag']
        }


@timeit
def compute_features(audio_path, params=PROFILE):
    """"""
    feature = AudioFeatures(audio_file=audio_path, sample_rate=params['sample_rate'])
    if feature.audio_vector.shape[0] == 0:
        raise IOError("Empty or invalid audio recording file -%s-" % audio_path)

    if params['endtime']:
        feature.audio_vector = feature.audio_slicer(endTime=params['endtime'])
    if params['downsample_audio']:
        feature.audio_vector = feature.resample_audio(params['sample_rate'] / params['downsample_factor'])

    out_dict = dict()
    # now we compute all the listed features in the profile dict and store the results to a output dictionary
    for method in params['features']:
        out_dict[method] = getattr(feature, method)()
    return out_dict


def compute_features_from_list_file(input_txt_file, feature_dir, params=PROFILE):
    """Compute certain audio features for a list of audio file paths"""

    start_time = time.time()
    _LOG_FILE.info("\nExtracting features for %s " % input_txt_file)
    data = read_txt_file(input_txt_file)
    data = [path for path in data if os.path.exists(path)]
    if len(data) <= 1:
        _LOG_FILE.debug("Empty collection txt file -%s- !" % input_txt_file)
        raise IOError("Empty collection txt file -%s- !" % input_txt_file)

    for song in data:
        feature_dict = compute_features(audio_path=song, params=params)
        track_id = os.path.basename(song).replace('.mp3', '')
        feature_dict['track_id'] = track_id
        # save as json
        with open(feature_dir + os.path.basename(input_txt_file) + '.json', 'w') as f:
            json.dump(feature_dict, f)
    _LOG_FILE.debug("Process finished in - %s - seconds" % (start_time - time.time()))


@timeit
def batch_feature_extractor(collections_dir, feature_dir, n_threads, params=PROFILE):
    """
    Compute parallelised feature extraction process from a collection of input audio file path txt files

    :param
        collections_dir: path to directory where a group of collections.txt located with list of audio file paths
        feature_dir: path where the computed audio features should be stored
        n_threads: no. of threads for parallelisation
        params: profile dict with params

    :return: None

    eg:
        '>>> batch_hpcp_extractor(collections_dir='./collections_dir/', features_dir='./features/', nthreads=4)'


        Here the collections_dir directory should have the following directory structure. Each *_collections.txt files
        contains a list of path to audio files in the disk for feature computation.

       ./collections_dir/
            /1_collections.txt
            /2_collections.txt
            .......
            .......
            /*_collections.txt

    """
    collection_files = glob.glob(collections_dir + '*.txt')
    feature_path = [feature_dir for i in range(len(collection_files))]
    param_list = [params for i in range(len(collection_files))]
    args = zip(collection_files, feature_path, param_list)

    Parallel(n_jobs=n_threads, verbose=1)(delayed(compute_features_from_list_file)\
                                              (cpath, fpath, param) for cpath, fpath, param in args)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "With command-line args, it does batch feature extraction of  \
            collection of audio files using multiple threads", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", action="store",
                        help="path to collection_files for audio feature extraction")
    parser.add_argument("-p", action="store",
                        help="path to directory where the audio features should be stored")
    parser.add_argument("-n", action="store", default=-1,
                        help="No of threads required for parallelization")

    cmd_args = parser.parse_args()

    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
    except ImportError:
        raise ImportError("Cannot find multiprocessing package on the system")

    if cores not in [0, 1]:
        batch_feature_extractor(cmd_args.c, cmd_args.p, cores-1)
    else:
        batch_feature_extractor(cmd_args.c, cmd_args.p, cmd_args.n)
    print ("... Done ....")

