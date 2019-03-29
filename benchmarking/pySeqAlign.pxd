from libcpp.vector cimport vector

cdef extern from "SequenceAlignment.c":
	double swalignimpconstrained(float* S, int N, int M)
