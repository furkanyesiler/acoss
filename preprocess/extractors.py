# -*- coding: utf-8 -*-
"""
Batch feature extractor

@2019
"""
from .utils import timeit, log, read_txt_file, ErrorFile
from .features import AudioFeatures
from joblib import Parallel, delayed
import local_config
import argparse
import datetime
import time
import glob
import json
import os


_TIMESTAMP = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
_LOG_FILE = log(_TIMESTAMP + "_extractor.log")
_ERROR_FILE = ErrorFile(_TIMESTAMP + "_errors.txt")


PROFILE = {
           'sample_rate': 44100,
           'input_audio_format': '.mp3',
           'downsample_audio': False,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['cqt', 
                        'hpcp', 
                        'crema', 
                        'chroma_cens',
                        'key_extractor',
                        'tempogram', 
                        'mfcc_htk']
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

    track_id = os.path.basename(audio_path).replace(params['input_audio_format'], '')
    out_dict['track_id'] = track_id

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
        try:
            feature_dict = compute_features(audio_path=song, params=params)
            # save as json
            with open(feature_dir + os.path.basename(song).replace(params['input_audio_format'], '') + '.json', 'w') as f:
                json.dump(feature_dict, f)
        except:
            _ERROR_FILE.add(song)
            _LOG_FILE.debug("Error: skipping computing features for audio file --%s-- " % song)
    _LOG_FILE.info("Process finished in - %s - seconds" % (start_time - time.time()))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "With command-line args, it does batch feature extraction of  \
            collection of audio files using multiple threads", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", action="store",
                        help="path to collection_files for audio feature extraction", default=local_config.COLLECTION_DIR)
    parser.add_argument("-p", action="store",
                        help="path to directory where the audio features should be stored", default=local_config.FEATURES_DIR)
    parser.add_argument("-n", action="store", default=-1,
                        help="No of threads required for parallelization")

    cmd_args = parser.parse_args()
    # here we can choose subset should be used
    local_config.create_benchmark_file(n_splits=100)
    if not os.path.exists(cmd_args.p):
        os.mkdir(cmd_args.p)
    # then do batch feature extraction with default params
    batch_feature_extractor(cmd_args.c, cmd_args.p, cmd_args.n)

    _ERROR_FILE.close()
    print ("... Done ....")
    print (" -- PROFILE INFO -- \n %s" % PROFILE)
