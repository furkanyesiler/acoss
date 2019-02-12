# -*- coding: utf-8 -*-
"""
Some utility functions

"""
import subprocess
import logging
import time


def log(log_file):
    """Returns a logger object with predefined settings"""
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
            print '%r - runtime : %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed


def read_txt_file(txt_file):
    """read a text file and strips \n char from it"""
    f = open(txt_file)
    data = f.readlines()
    return [d.replace('\n', '') for d in data]


def savelist_to_file(pathList, filename):
    doc = open(filename, 'w')
    for item in pathList:
        doc.write("%s\n" % item)
    doc.close()
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
