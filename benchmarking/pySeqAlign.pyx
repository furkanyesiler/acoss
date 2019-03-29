cimport numpy as np
cimport pySeqAlign as seq
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def swconstrained(np.ndarray[float,ndim=1,mode="c"] SParam not None, int N, int M):

	res = seq.swalignimpconstrained(&SParam[0], N, M)

	return res