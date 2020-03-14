# -*- coding: utf-8 -*-
"""
Module containing various algorithms for cover song identification task benchmarking.
"""
from .algorithm_template import *
from .earlyfusion_traile import *
from .latefusion_chen import *
from .rqa_serra09 import *
from .simple_silva import *
from .ftm2d import *

__all__ = [_ for _ in dir() if not _.startswith('_')]
