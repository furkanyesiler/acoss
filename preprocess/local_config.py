# -*- coding: utf-8 -*-
"""
@2019
"""
import utils as util
import os


PROJECT_PATH = "/datasets/MTG/projects/tacos/"
SONG_DIR = PROJECT_PATH + "downloads/songs/"
FEATURE_DIR = PROJECT_PATH + "features/"
MODULE_PATH = PROJECT_PATH + "coversongdataset/"
SUBSETS_DIR = MODULE_PATH + "subsets/"
COLLECTION_DIR = SUBSETS_DIR + "collection_txts_furkan/" 
BENCHMARK_SUBSET_FILE = SUBSETS_DIR + "benchmark_subset_paths_v2.txt"
WHATISACOVER_SUBSET_FILE = SUBSETS_DIR + "whatisacover_subset_paths_v2.txt"
LOG_PATH = MODULE_PATH + "preprocess/logs/"


if not os.path.exists(LOG_PATH):
  os.mkdir(LOG_PATH)


def create_benchmark_files(n_splits):
  """"""
  paths = util.read_txt_file(BENCHMARK_SUBSET_FILE)
  paths = [SONG_DIR + path for path in paths]
  os.system("rm -r -f %s" % COLLECTION_DIR)
  create_collection_files(paths, n_splits=n_splits)


def create_whatisacover_files(n_splits):
  """"""
  paths = util.read_txt_file(WHATISACOVER_SUBSET_FILE)
  paths = [SONG_DIR + path for path in paths]
  os.system("rm -r -f %s" % COLLECTION_DIR)
  create_collection_files(paths, n_splits=n_splits)


def create_collection_files(paths, n_splits=100):
  """"""
  import numpy as np
  for path in paths:
    if not os.path.exists(path):
      raise Exception(".. Invalid audio filepath -- %s -- found in the collection file" % path)
  song_chunks = np.array_split(paths, n_splits)
  if not os.path.exists(COLLECTION_DIR):
    os.mkdir(COLLECTION_DIR)
  for idx, chunk in enumerate(song_chunks):
    util.savelist_to_file(chunk, COLLECTION_DIR + str(idx) + '_collections.txt')


