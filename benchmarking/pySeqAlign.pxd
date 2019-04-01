from libcpp.vector cimport vector

cdef extern from "SequenceAlignment.c":
	double swalignimpconstrained(unsigned char* S, float* D, int N, int M)

cdef extern from "SequenceAlignment.c":
	double qmax_c(unsigned char* S, float* D, int N, int M)

cdef extern from "SequenceAlignment.c":
	double dmax_c(unsigned char* S, float* D, int N, int M)