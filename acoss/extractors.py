# -*- coding: utf-8 -*-
"""
Batch feature extractor

@2019
"""
import argparse
import datetime
import time
import glob
import os
import deepdish as dd
from joblib import Parallel, delayed

from .utils import log, read_txt_file, savelist_to_file, create_audio_path_batches
from .features import AudioFeatures
from .local_config import *


PROFILE = {
           'sample_rate': 44100,
           'input_audio_format': '.mp3',
           'downsample_audio': False,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['hpcp',
                        'key_extractor',
                        'tempogram',
                        'madmom_features',
                        'mfcc_htk']
        }


_TIMESTAMP = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
_LOG_PATH = "./logs/"
if not os.path.exists(_LOG_PATH):
    os.makedirs(_LOG_PATH)
_LOG_FILE = log(_LOG_PATH + _TIMESTAMP + "_extractor.log")
_ERRORS = list()


def compute_features(audio_path, params=PROFILE):
    """
    Compute a list of audio features for a given audio file as per the extractor profile.

    NOTE: Audio files should be structured in a way that each cover song clique has a folder with it's tracks inside to
          have the correct cover label in the resulted feature dictionary.

          eg: ./audio_dir/
                    /cover_clique_label/
                        /audio_file.mp3

    :param audio_path: path to audio file
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params)

    :return: a dictionary with all the requested features computes as key, value pairs.
    """
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

    track_id = os.path.basename(audio_path).replace(params['input_audio_format'], '')
    out_dict['track_id'] = track_id

    label = audio_path.split('/')[-2]
    out_dict['label'] = label

    return out_dict


def compute_features_from_list_file(input_txt_file, feature_dir, params=PROFILE):
    """
    Compute specified audio features for a list of audio file paths and store to disk as .h5 file
    from a given input text file.
    It is a wrapper around 'compute_features'.

    :param input_txt_file: a text file with a list of audio file paths
    :param feature_dir: a path
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params)

    :return: None
    """

    start_time = time.time()
    _LOG_FILE.info("\nExtracting features for %s " % input_txt_file)
    data = read_txt_file(input_txt_file)
    data = [path for path in data if os.path.exists(path)]
    if len(data) < 1:
        _LOG_FILE.debug("Empty collection txt file -%s- !" % input_txt_file)
        raise IOError("Empty collection txt file -%s- !" % input_txt_file)

    for song in data:
        try:
            feature_dict = compute_features(audio_path=song, params=params)
            # save as h5
            dd.io.save(feature_dir + os.path.basename(song).replace(params['input_audio_format'], '') + '.h5',
                       feature_dict)
        except:
            _ERRORS.append(input_txt_file)
            _ERRORS.append(song)
            _LOG_FILE.debug("Error: skipping computing features for audio file --%s-- " % song)

    _LOG_FILE.info("Process finished in - %s - seconds" % (start_time - time.time()))


def batch_feature_extractor(collections_dir, feature_dir, n_threads, mode='parallel', params=PROFILE):
    """
    Compute parallelised feature extraction process from a collection of input audio file path txt files

    :param
        collections_dir: path to directory where a group of collections.txt located with list of audio file paths
        feature_dir: path where the computed audio features should be stored
        n_threads: no. of threads for parallelisation
        mode: whether to run the extractor in 'single' or 'parallel' mode.
        params: profile dict with params

    :return: None

    eg:
        '>>> batch_hpcp_extractor(collections_dir='./collections_dir/', features_dir='./features/', n_threads=4)'


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
    print("Computing batch feature extraction using '%s' mode the profile: %s " % (mode, params))
    if mode == 'parallel':
        Parallel(n_jobs=n_threads, verbose=1)(delayed(compute_features_from_list_file)\
                                              (cpath, fpath, param) for cpath, fpath, param in args)
    elif mode == 'single':
        tic = time.monotonic()
        for cpath, fpath, param in args:
            compute_features_from_list_file(cpath, fpath, param)
        print("Single mode feature extraction finished in %s" % (time.monotonic() - tic))
    else:
        raise IOError("Wrong value for the parameter 'mode'. Should be either 'single' or 'parallel'")
    savelist_to_file(_ERRORS, _LOG_PATH + _TIMESTAMP + '_erros_extractor.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="With command-line args, it does batch feature extraction of  \
            collection of audio files using multiple threads", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--collection_dir", action="store",
                        help="path to collection_file for audio feature extraction", default=COLLECTION_DIR)
    parser.add_argument("-p", "--feature_dir", action="store",
                        help="path to directory where the audio features should be stored", default=FEATURE_DIR)
    parser.add_argument("-f", "--feature_list", action="store", type=str, default="['hpcp', 'key_extractor', "
                                                                                  "'crema', 'madmom_features', "
                                                                                  "'mfcc_htk']",
                        help="List of features to compute. Eg. ['hpcp' 'crema']")
    parser.add_argument("-m", "--run_mode", action="store", default='parallel',
                        help="Whether to run the extractor in single or parallel mode. "
                             "Choose one of ['single', 'parallel']")
    parser.add_argument("-d", "--dataset_csv", action="store", default='',
                        help="path to input dataset csv file")
    parser.add_argument("-n", "--workers", action="store", default=-1,
                        help="No of workers for running the batch extraction process. Only valid in 'parallel' mode.")

    cmd_args = parser.parse_args()

    print("Args: %s" % cmd_args)

    if not os.path.exists(cmd_args.p):
        os.mkdir(cmd_args.p)

    batch_feature_extractor(cmd_args.c, cmd_args.p, cmd_args.n, cmd_args.m)

    print("... Done ....")
    print(" -- PROFILE INFO -- \n %s" % PROFILE)
