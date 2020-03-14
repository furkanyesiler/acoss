# test if the feature extractors functions are properly imported
from acoss.utils import COVERS_80_CSV
from acoss.extractors import batch_feature_extractor
from acoss.extractors import PROFILE

# test if the benchmark interface is propoerly imported
from acoss.coverid import benchmark, algorithm_names
from acoss.utils import COVERS_80_CSV

# test if numba functions are properly imported
from acoss.algorithms.utils.alignment_tools import smith_waterman_constrained

print("acoss properly imported")


# TODO: write proper unit tests

from unittest import TestCase

class AcossTest(TestCase):
    pass
