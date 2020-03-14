# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

__all__ = ['smith_waterman_constrained']

@jit(nopython=True)
def delta_func(value_a, value_b, gap_opening=-0.5, gap_extension=-0.7):
    if value_a > 0:
        return 0
    elif value_b == 0 and value_a > 0:
        return gap_opening
    else:
        return gap_extension

@jit(nopython=True)
def match(value, match_score=1, mismatch_score=-1):
    if value == 0:
        return mismatch_score
    elif value == 1:
        return match_score
    else:
        raise IOError("Non-binary elements found in input")

@jit(nopython=True)
def smith_waterman_constrained(input_matrix):
    """
    https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
    
    Constrained alignments according to [1]
    [1] Tralie, C.J., 2017. Early mfcc and hpcp fusion for robust cover song identification. arXiv preprint arXiv:1707.04680."""
    max_score = 0.0
    M, N = input_matrix.shape
    if N < 4 or M < 4:
        return max_score
    score_matrix = np.zeros((M, N))
    for i in range(3, M):
        for j in range(3, N):
            match_value = match(input_matrix[i-1][j-1])
            d1 = score_matrix[i-1][j-1] + match_value + delta_func(input_matrix[i-2][j-2], input_matrix[i-1][j-1])
            d2 = score_matrix[i-2][j-1] + match_value + delta_func(input_matrix[i-3][j-2], input_matrix[i-1][j-1])
            d3 = score_matrix[i-1][j-2] + match_value + delta_func(input_matrix[i-2][j-3], input_matrix[i-1][j-1])
            score_matrix[i][j] = max([d1, d2, d3, 0.0])
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]    
    return max_score
