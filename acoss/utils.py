# -*- coding: utf-8 -*-
"""
Some general utility functions

"""
import datetime
import logging
import time
import os
import numpy as np
from shutil import rmtree

_LOG_PATH = "./logs/"
_TIMESTAMP = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())


def log(log_file):
    """Returns a logger object with predefined settings"""
    log_file = _LOG_PATH + _TIMESTAMP + "_" + log_file
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


def create_audio_path_batches(dataset_csv, dir_to_save, batch_size=50, root_audio_dir="./", audio_format="mp3", reset=False):

    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)
    elif reset:
        rmtree(dir_to_save)

    data_paths = create_dataset_filepaths(dataset_csv, root_audio_dir, audio_format)
    for path in data_paths:
        if not os.path.exists(path):
            raise Exception(".. Invalid audio filepath -- %s -- found in the collection file" % path)
    if batch_size > len(data_paths):
        raise UserWarning("Batch size shouldn't be greater than the size of audio file")
    song_chunks = np.array_split(data_paths, batch_size)
    for idx, chunk in enumerate(song_chunks):
        savelist_to_file(chunk, dir_to_save + str(idx) + '_th_batch.txt')


def create_dataset_filepaths(dataset_csv, root_audio_dir="./", file_format=".mp3"):
    import pandas as pd
    dataset = pd.read_csv(dataset_csv)
    dataset['filepath'] = dataset.apply(lambda x: root_audio_dir + x.work_id + "/" + x.track_id + file_format, axis=1)
    return dataset.filepath.tolist()
