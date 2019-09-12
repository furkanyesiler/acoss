# -*- coding: utf-8 -*-
"""

@2019
"""


def benchmark(dataset, feature_type="hpcp", method="serra09", short_name="covers80"):
    """A wrapper function to run specific benchmark cover id algorithms (TODO)"""
    if method == "Serra09":
        from .benchmark.rqa_serra09 import Serra09
        # here run the algo
        pass
    elif method == "EarlyFusion":
        from .benchmark.earlyfusion_traile import EarlyFusion
    elif method == "ChenFusion":
        from .benchmark.latefusion_chen import ChenFusion
    elif method == "FTM2D":
        from .benchmark.ftm2d import FTM2D
    elif method == "Simple":
        from .benchmark.simple_silva import Simple
    else:
        raise NotImplementedError("Cannot find the inputted method in the benchmark algorithm lists")
    return
