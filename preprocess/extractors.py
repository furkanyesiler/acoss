# -*- coding: utf-8 -*-
"""
[TODO]: Here we can add batch feature extractors for all audio files in the dataset with multiple threads

The following functions are some examples and we can refactor it to meet our needs (add more features or store in more optimised fileformats etc)

"""
from joblib import Parallel, delayed
from .features import AudioFeatures
from .utils import timeit, log
import numpy as np
import time
import glob


_LOG_FILE = log("<path_to_log_file>")


PROFILE = {
            'sample_rate': 44100,
            'hop_size': 2048,
            'frame_size': 4096,
            'num_bins': 12,
            'max_peaks': 100,
            'min_frequency': 100,
            'max_frequency': 3500
        }


def compute_hpcp(audio_path, downsample=True, downsample_factor=2, endTime=None, params=PROFILE):
    """Compute hpcp feature array for a input audio file path"""
    feature = AudioFeatures(audio_file=audio_path, sample_rate=params['sample_rate'])
    if feature.audio_vector.shape[0] == 0:
        raise IOError("Empty or invalid audio recording file -%s-" % audio_path)
    if endTime:
        feature.audio_vector = chroma.audio_slicer(endTime=endTime)
    if downsample:
        feature.audio_vector = chroma.resample_audio(params['sample_rate'] / downsample_factor)

    hpcp = feature.chroma_hpcp(hopSize=params['hop_size'],
                              frameSize=params['frame_size'],
                              maxPeaks=params['max_peaks'],
                              numBins=params['num_bins'],
                              minFrequency=params['min_frequency'],
                              maxFrequency=params['max_frequency'])
    return hpcp


@timeit
def compute_hpcp_from_list_file(input_txt_file,
                                feature_dir,
                                downsample=False,
                                downsample_factor=2,
                                endTime=None,
                                params=PROFILE):
    """Compute hpcp features for a list of audio file paths"""
    start_time = time.time()
    _LOG_FILE.info("\nExtracting features for %s " % input_txt_file)
    data = txt_utils.read_txt_file(input_txt_file)
    data = [path for path in data if os.path.exists(path)]
    if len(data) <= 1:
        _LOG_FILE.debug("Empty collection txt file -%s- !" % input_txt_file)
        raise IOError("Empty collection txt file -%s- !" % input_txt_file)
    for song in data:
        hpcp = compute_hpcp(audio_path=song,
                            params=params,
                            downsample=downsample,
                            downsample_factor=downsample_factor,
                            endTime=endTime)
        audio_id = os.path.basename(song).replace('.mp3', '')
        np.save(feature_dir + audio_id + '.npy', hpcp)
    _LOG_FILE.debug("Process finished in - %s - seconds" % (start_time - time.time()))


def batch_hpcp_extractor(collections_dir,
                         feature_dir,
                         n_threads,
                         downsample=False,
                         downsample_factor=2,
                         endTime=None,
                         params=PROFILE):
    """
    Compute parallelised hpcp feature extraction process from a collection of input audio file path txt files

    :param
        collections_dir: path to directory where a group of collections.txt located with list of audio file paths
        feature_dir: path where the computed audio features should be stored
        n_threads: no. of threads for parallelisation
        params: profile dict with params
        endTime: Slice audio with a given end time before feature extraction

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
    end_times = [endTime for i in range(len(collection_files))]
    dstate = [downsample for i in range(len(collection_files))]
    dfactor = [downsample_factor for i in range(len(collection_files))]
    args = zip(collection_files, feature_path, dstate, dfactor, end_times, param_list)

    Parallel(n_jobs=n_threads, verbose=1)(delayed(compute_hpcp_from_list_file)
                                (cpath, fpath, d, step, etime, param) for cpath, fpath, d, step, etime, param in args)
    return
