cimport numpy as np
cimport acoss.benchmark.utils.alignment_tools.pySeqAlign as seq
#cimport pySeqAlign as seq
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def swconstrained(np.ndarray[unsigned char,ndim=1,mode="c"] SParam not None, np.ndarray[float,ndim=1,mode="c"] DParam not None, int N, int M):
	res = seq.swalignimpconstrained(&SParam[0], &DParam[0], N, M)
	return res


@cython.boundscheck(False)
@cython.wraparound(False)
def qmax(np.ndarray[unsigned char,ndim=1,mode="c"] SParam not None, np.ndarray[float,ndim=1,mode="c"] DParam not None, int N, int M):
	res = seq.qmax_c(&SParam[0], &DParam[0], N, M)
	return res


@cython.boundscheck(False)
@cython.wraparound(False)
def dmax(np.ndarray[unsigned char,ndim=1,mode="c"] SParam not None, np.ndarray[float,ndim=1,mode="c"] DParam not None, int N, int M):
	res = seq.dmax_c(&SParam[0], &DParam[0], N, M)
	return res