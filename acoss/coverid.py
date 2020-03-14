# -*- coding: utf-8 -*-
"""
Interface to run the various cover id algorithms for acoss benchmarking.
"""
import argparse
import time
import sys
import os
from shutil import rmtree

from .utils import log

__all__ = ['benchmark', 'algorithm_names']

_LOG_FILE_PATH = "acoss.coverid.log"
_LOGGER = log(_LOG_FILE_PATH)

# list the available cover song identification algorithms in acoss
algorithm_names = ["Serra09", "EarlyFusionTraile", "LateFusionChen", "FTM2D", "SiMPle"]


def benchmark(dataset_csv,
              feature_dir,
              feature_type="hpcp",
              algorithm="Serra09",
              shortname="covers80",
              parallel=True,
              n_workers=-1):
    """Benchmark a specific cover id algorithm with a given input dataset annotation csv file.
    
    Arguments:
        dataset_csv {string} -- path to dataset csv annotation file
        feature_dir {string} -- path to the directory where the pre-computed audio features are stored
    
    Keyword Arguments:
        feature_type {str} -- type of audio feature you want to use for benchmarking. (default: {"hpcp"})
        algorithm {str} -- name of the algorithm you want to benchmark (default: {"Serra09"})
        shortname {str} -- description (default: {"DaTacos-Benchmark"})
        parallel {bool} -- whether you want to run the benchmark process with parallel workers (default: {True})
        n_workers {int} -- number of workers required. By default it uses as much workers available on the system. (default: {-1})
    
    Raises:
        NotImplementedError: when an given algorithm method in not implemented in acoss.benchmark
    """

    if algorithm not in algorithm_names:
        warn = ("acoss.coverid: Couldn't find '%s' algorithm in acoss \
                                Available cover id algorithms are %s "
                                % (algorithm, str(algorithm_names)))
        _LOGGER.debug(warn)
        raise NotImplementedError(warn)

    _LOGGER.info("Running acoss cover identification benchmarking for the algorithm - '%s'" % algorithm)

    start_time = time.monotonic()

    if algorithm == "Serra09":
        from .algorithms.rqa_serra09 import Serra09
        # here run the algo
        serra09 = Serra09(dataset_csv=dataset_csv,
                          datapath=feature_dir,
                          chroma_type=feature_type,
                          shortname=shortname)
        _LOGGER.info('Computing pairwise similarity...')
        serra09.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        serra09.normalize_by_length()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in serra09.Ds.keys():
            serra09.getEvalStatistics(similarity_type)
        serra09.cleanup_memmap()

    elif algorithm == "EarlyFusion":
        from .algorithms.earlyfusion_traile import EarlyFusion

        early_fusion = EarlyFusion(dataset_csv=dataset_csv,
                                   datapath=feature_dir,
                                   chroma_type=feature_type,
                                   shortname=shortname)
        _LOGGER.info('Feature loading done...')
        for i in range(len(early_fusion.filepaths)):
            early_fusion.load_features(i)
        _LOGGER.info('Computing pairwise similarity...')
        early_fusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        early_fusion.do_late_fusion()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in early_fusion.Ds:
            early_fusion.getEvalStatistics(similarity_type)
        early_fusion.cleanup_memmap()

    elif algorithm  == "LateFusionChen":
        from .algorithms.latefusion_chen import ChenFusion

        chenFusion = ChenFusion(dataset_csv=dataset_csv,
                                datapath=feature_dir,
                                chroma_type=feature_type,
                                shortname=shortname)
        _LOGGER.info('Computing pairwise similarity...')
        chenFusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        chenFusion.normalize_by_length()
        chenFusion.do_late_fusion()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in chenFusion.Ds.keys():
            _LOGGER.info(similarity_type)
            chenFusion.getEvalStatistics(similarity_type)
        chenFusion.cleanup_memmap()

    elif algorithm == "FTM2D":
        from .algorithms.ftm2d import FTM2D

        ftm2d = FTM2D(dataset_csv=dataset_csv,
                      datapath=feature_dir,
                      chroma_type=feature_type,
                      shortname=shortname)
        for i in range(len(ftm2d.filepaths)):
            ftm2d.load_features(i)
        _LOGGER.info('Feature loading done...')
        _LOGGER.info('Computing pairwise similarity...')
        ftm2d.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in ftm2d.Ds.keys():
            ftm2d.getEvalStatistics(similarity_type)
        ftm2d.cleanup_memmap()

    elif algorithm == "SiMPle":
        from .algorithms.simple_silva import Simple

        simple = Simple(dataset_csv=dataset_csv,
                        datapath=feature_dir,
                        chroma_type=feature_type,
                        shortname=shortname)
        for i in range(len(simple.filepaths)):
            simple.load_features(i)
        _LOGGER.info('Feature loading done...')
        _LOGGER.info('Computing pairwise similarity...')
        simple.all_pairwise(parallel, n_cores=n_workers, symmetric=False)
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in simple.Ds.keys():
            simple.getEvalStatistics(similarity_type)
        simple.cleanup_memmap()

    _LOGGER.info("acoss.coverid benchmarking finsihed in %s" % (time.monotonic() - start_time))
    _LOGGER.info("Log file located at '%s'" % _LOG_FILE_PATH)


def parser_args(cmd_args):

    parser = argparse.ArgumentParser(sys.argv[0], description="Benchmark a specific cover id algorithm with a given"
                                                              "input dataset csv annotations",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '--dataset_csv', type=str, action="store",
                        help="Input dataset csv file")
    parser.add_argument("-d", '--feature_dir', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-m", "--method", type=str, action="store", default="covers80",
                        help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default="hpcp",
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=1,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_workers', type=int, action="store", default=-1,
                        help="No of workers required for parallelization")

    return parser.parse_args(cmd_args)


if __name__ == '__main__':

    args = parser_args(sys.argv[1:])

    benchmark(dataset_csv=args.i,
              feature_dir=args.d,
              feature_type=args.c,
              method=args.m,
              parallel=bool(args.p),
              n_workers=args.m)
