# -*- coding: utf-8 -*-
"""
Some utility functions

"""
import subprocess
import datetime
import logging
import time
import os

import numpy as np

from .local_config import *

_TIMESTAMP = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())


def log(log_file):
    """Returns a logger object with predefined settings"""
    log_file = LOG_PATH + _TIMESTAMP + "_" + log_file
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger


def timeit(method):
    """A custom timeit profiling decorator from Stackoverflow"""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r - runtime : %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def read_txt_file(txt_file):
    """read a text file and strips \n char from it"""
    f = open(txt_file)
    data = f.readlines()
    return [d.replace('\n', '') for d in data]


def savelist_to_file(path_list, filename):
    doc = open(filename, 'w')
    for item in path_list:
        doc.write("%s\n" % item)
    doc.close()


def create_benchmark_files(n_splits):
    paths = read_txt_file(BENCHMARK_SUBSET_FILE)
    paths = [SONG_DIR + path for path in paths]
    os.system("rm -r -f %s" % COLLECTION_DIR)
    create_collection_files(paths, n_splits=n_splits)


def create_whatisacover_files(n_splits):
    paths = read_txt_file(WHATISACOVER_SUBSET_FILE)
    paths = [SONG_DIR + path for path in paths]
    os.system("rm -r -f %s" % COLLECTION_DIR)
    create_collection_files(paths, n_splits=n_splits)


def create_collection_files(paths, dir_to_save=COLLECTION_DIR, n_splits=100):
    import numpy as np
    for path in paths:
        if not os.path.exists(path):
            raise Exception(".. Invalid audio filepath -- %s -- found in the collection file" % path)
    song_chunks = np.array_split(paths, n_splits)
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)
    for idx, chunk in enumerate(song_chunks):
        savelist_to_file(chunk, dir_to_save + str(idx) + '_collections.txt')


def create_audio_path_batches(input_txt_file, dir_to_save=COLLECTION_DIR, batch_size=50):

    return


def ffmpeg_slicer(filename, start_time, end_time, out_filename):
    """
    Description:
                Slice a audio at specified start and end time and save it as a new audio file

    :Params:
            filename: path to a input audio file
            start_time: desired start time for slicing
            end_time: desired end time for slicing
            out_filename: path to the desired output audio file
                          [Note: The fileformat of input and output audiofile should be same otherwise
                          ffmpeg throws an error]

    NOTE: needs ffmpeg locally installed

    """
    return subprocess.call('ffmpeg -i %s -acodec copy -ss %s -to %s %s'
                            % (filename, start_time, end_time, out_filename), shell=True)


class ErrorFile(object):

    def __init__(self, filename):
        filename = LOG_PATH + _TIMESTAMP + "_" + filename
        self.doc = open(filename, 'w')
        self.errors = list()
        self.doc.write("---")

    def add(self, text):
        self.errors.append(text)
        self.doc.write("%s\n" % text)

    def close(self):
        self.doc.close()


def global_hpcp(chroma):
    """Computes global hpcp of a input chroma vector"""
    if chroma.shape[1] not in [12, 24, 36]:
        raise IOError("Wrong axis for the input chroma array. Expected shape '(frame_size, bin_size)'")
    return np.divide(chroma.sum(axis=0), np.max(chroma.sum(axis=0)))


def optimal_transposition_index(chromaA, chromaB, n_shifts=12):
    """
    Computes optimal transposition index (OTI) for the chromaB to be transposed in the same key as of chromaA
    Input :
            chromaA : chroma feature array of the query song for reference key
            chromaB : chroma feature array of the reference song for which OTI has to be applied
        Params:
                n_shifts: (default=12) Number of oti tranpositions to be checked for circular shift
    Output : Integer value specifying optimal transposition index for transposing chromaB to chromaA to be in same key
    """
    global_hpcpA = global_hpcp(chromaA)
    global_hpcpB = global_hpcp(chromaB)
    idx = list()
    for index in range(n_shifts):
        idx.append(np.dot(global_hpcpA, np.roll(global_hpcpB, index)))
    return int(np.argmax(idx))


def transpose_by_oti(chromaB, oti=0):
    """
    Transpose the chromaB vector to a common key by a value of optimal transposition index
    Input :
            chromaB : input chroma array
    Output : chromaB vector transposed to a factor of specified OTI
    """
    return np.roll(chromaB, oti)