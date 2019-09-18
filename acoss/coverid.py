# -*- coding: utf-8 -*-
"""

@2019
"""
import argparse
import sys


def benchmark(dataset_csv,
              feature_dir,
              feature_type="hpcp",
              method="serra09",
              shortname="covers80",
              parallel=True,
              n_cores=-1):
    """A wrapper function to run specific benchmark cover id algorithms (TODO)"""

    if method == "Serra09":
        from .benchmark.rqa_serra09 import Serra09
        # here run the algo
        serra09 = Serra09(dataset_csv=dataset_csv,
                          datapath=feature_dir,
                          chroma_type=feature_type,
                          shortname=shortname)
        serra09.all_pairwise(parallel, n_cores, symmetric=True)
        serra09.normalize_by_length()
        for similarity_type in serra09.Ds.keys():
            print(similarity_type)
            serra09.getEvalStatistics(similarity_type)
        serra09.cleanup_memmap()

    elif method == "EarlyFusion":
        from .benchmark.earlyfusion_traile import EarlyFusion

        early_fusion = EarlyFusion(dataset_csv=dataset_csv,
                                   datapath=feature_dir,
                                   chroma_type=feature_type,
                                   shortname=shortname)
        for i in range(len(early_fusion.filepaths)):
            print("Preloading features %i of %i" % (i + 1, len(early_fusion.filepaths)))
            early_fusion.load_features(i)
        early_fusion.all_pairwise(parallel, n_cores, symmetric=True)
        early_fusion.do_late_fusion()
        for similarity_type in early_fusion.Ds:
            early_fusion.getEvalStatistics(similarity_type)
        early_fusion.cleanup_memmap()

    elif method == "ChenFusion":
        from .benchmark.latefusion_chen import ChenFusion

        chenFusion = ChenFusion(dataset_csv=dataset_csv,
                                datapath=feature_dir,
                                chroma_type=feature_type,
                                shortname=shortname)
        chenFusion.all_pairwise(parallel, n_cores, symmetric=True)
        chenFusion.normalize_by_length()
        chenFusion.do_late_fusion()
        for similarity_type in chenFusion.Ds.keys():
            print(similarity_type)
            chenFusion.getEvalStatistics(similarity_type)
        chenFusion.cleanup_memmap()

    elif method == "FTM2D":
        from .benchmark.ftm2d import FTM2D

        ftm2d = FTM2D(dataset_csv=dataset_csv,
                      datapath=feature_dir,
                      chroma_type=feature_type,
                      shortname=shortname)
        for i in range(len(ftm2d.filepaths)):
            ftm2d.load_features(i)
        print('Feature loading done.')
        ftm2d.all_pairwise(parallel, n_cores, symmetric=True)
        for similarity_type in ftm2d.Ds.keys():
            ftm2d.getEvalStatistics(similarity_type)
        ftm2d.cleanup_memmap()

    elif method == "Simple":
        from .benchmark.simple_silva import Simple

        simple = Simple(dataset_csv=dataset_csv,
                        datapath=feature_dir,
                        chroma_type=feature_type,
                        shortname=shortname)
        for i in range(len(simple.filepaths)):
            simple.load_features(i)
        print('Feature loading done.')
        simple.all_pairwise(parallel, n_cores, symmetric=False)
        for similarity_type in simple.Ds.keys():
            simple.getEvalStatistics(similarity_type)
        simple.cleanup_memmap()
    else:
        raise NotImplementedError("Cannot find the inputted method in the benchmark algorithm lists")
    return


def parser_args(cmd_args):

    parser = argparse.ArgumentParser(sys.argv[0], description="Benchmark a specific cover id algorithm with a given"
                                                              "input dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '--dataset_csv', type=str, action="store",
                        help="Input dataset csv file")
    parser.add_argument("-d", '--feature_dir', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-m", "--method", type=str, action="store", default="covers80",
                        help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default="hpcp",
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=-1,
                        help="No of cores required for parallelization")

    return parser.parse_args(cmd_args)


if __name__ == '__main__':

    args = parser_args(sys.argv[1:])

    benchmark(dataset_csv=args.i,
              feature_dir=args.d,
              feature_type=args.c,
              method=args.m,
              parallel=bool(args.p),
              n_cores=args.m)

