from libcpp.vector cimport vector

cdef extern from "SequenceAlignment.c":
	double swalignimpconstrained(unsigned char* S, int N, int M)
