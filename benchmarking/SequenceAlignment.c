/*Programmer: Chris Tralie
*Purpose: To implement an implicit version of Smith-Waterman that works on
*a binary dissimilarity matrix*/
#include <stdio.h>
#include <stdlib.h>

float tripleMax(float a, float b, float c) {
    float max = a;
    if (b > a) max = b;
    if (c > max) max = c;
    return max;
}

float quadMax(float a, float b, float c, float d) {
    float max = a;
    if (b > a) max = b;
    if (c > max) max = c;
    if (d > max) max = d;
    return max;
}

float fiveMax(float a, float b, float c, float d, float e) {
    float max = a;
    float t = quadMax(b, c, d, e);
    if (t > a) {
        return t;
    }
    return a;
}

float sixMax(float a, float b, float c, float d, float e, float f) {
    float t1 = tripleMax(a, b, c);
    float t2 = tripleMax(d, e, f);
    if (t1 > t2) {
        return t1;
    }
    return t2;
}




float Delta(float a, float b) {
    /*Parameters used in the paper*/
    float gapOpening = -0.5; 
    float gapExtension = -0.7;
    if (b > 0) {
        return 0;
    }
    if (b == 0 && a > 0) {
        return gapOpening;
    }
    return gapExtension;
}

float Match(unsigned char i) {
    float matchScore = 1;
    float mismatchScore = -1;
    if (i == 0) {
        return mismatchScore;
    }
    return matchScore;
}

/*Inputs: 
S (a binary N x M cross-similarity matrix)
D: A (N+1) x (M+1) dynamic programming matrix, starting
out as all 0s but which will hold the result
*/

/*Outputs: 1) Distance (scalar)
*2) (N+1) x (M+1) dynamic programming matrix (Optional)*/
float swalignimpconstrained(unsigned char* S, float* D, int N, int M) {
    size_t i, j;
    float maxD, d1, d2, d3, MS;
    
    N++; M++;
    if (N < 4 || M < 4) {
        return 0.0;
    }    

    maxD = 0.0;
    for (i = 3; i < (size_t)N; i++) {
        for (j = 3; j < (size_t)M; j++) {
            MS = Match(S[(i-1)*(M-1)+(j-1)]);
            /*H_(i-1, j-1) + S_(i-1, j-1) + delta(S_(i-2,j-2), S_(i-1, j-1))*/
            d1 = D[(i-1)*M+(j-1)] + MS + Delta(S[(i-2)*(M-1)+(j-2)], S[(i-1)*(M-1)+(j-1)]);
            /*H_(i-2, j-1) + S_(i-1, j-1) + delta(S_(i-3, j-2), S_(i-1, j-1))*/
            d2 = D[(i-2)*M+(j-1)] + MS + Delta(S[(i-3)*(M-1)+(j-2)], S[(i-1)*(M-1)+(j-1)]);
            /*H_(i-1, j-2) + S_(i-1, j-1) + delta(S_(i-2, j-3), S_(i-1, j-1))*/
            d3 = D[(i-1)*M+(j-2)] + MS + Delta(S[(i-2)*(M-1)+(j-3)], S[(i-1)*(M-1)+(j-1)]);
            D[i*M+j] = quadMax(d1, d2, d3, 0.0);
            if (D[i*M+j] > maxD) {
                maxD = D[i*M+j];
            }
        }
    }
    return maxD;
}




float gammaState(unsigned char value) {
    float onset = 0.5;
    float extension = 0.5;
    if (value == 1) {
        return onset;
    }
    return extension;
}

float qmax_c(unsigned char* S, float* D, int M, int N) {
    size_t i, j;
    float maxD, c1, c2, c3;
    
    if (N < 3 || M < 3) {
        return 0.0;
    }
    maxD = 0.0;
    for(i = 2; i < (size_t)M; i++) {
        for(j = 2; j < (size_t)N; j++) {
            // measure the diagonal when a similarity is found in the input matrix
            if (S[i*N+j] == 1) {
                c1 = D[(i-1)*N+j-1];
                c2 = D[(i-2)*N+j-1];
                c3 = D[(i-1)*N+j-2];
                D[i*N+j] = tripleMax(c1, c2, c3) + 1;
            }
            else {
            // apply gap penalty onset for disruption and extension when similarity is not found in the input matrix
                c1 = D[(i-1)*N+j-1] - gammaState(S[(i-1)*N+j-1]);
                c2 = D[(i-2)*N+j-1] - gammaState(S[(i-2)*N+j-1]);
                c3 = D[(i-1)*N+j-2] - gammaState(S[(i-1)*N+j-2]);
                D[i*N+j] = quadMax(c1, c2, c3, 0.0);
            }
            if (D[i*N+j] > maxD) {
                maxD = D[i*N+j];
            }
        }
    }
    return maxD;
}



float dmax_c(unsigned char* S, float* D, int M, int N) {
    size_t i, j;
    float maxD, c1, c2, c3, c4, c5;
    
    if (N < 4 || M < 4) {
        return 0.0;
    }
    maxD = 0.0;
    for(i = 3; i < (size_t)M; i++) {
        for(j = 3; j < (size_t)N; j++) {
            // measure the diagonal when a similarity is found in the input matrix
            if (S[i*N+j] == 1) {
                c2 = D[(i-2)*N+j-1] + S[(i-1)*N+j];
                c3 = D[(i-1)*N+j-2] + S[i*N+j-1];
                c4 = D[(i-3)*N+j-1] + S[(i-2)*N+j] + S[(i-1)*N+j];
                c5 = D[(i-1)*N+j-3] + S[i*N+j-2] + S[i*N+j-1];
                D[i*N+j] = fiveMax(D[(i-1)*N+j-1], c2, c3, c4, c5) + 1;
            }
            else {
            // apply gap penalty onset for disruption and extension when similarity is not found in the input matrix
                c1 = D[(i-1)*N+j-1] - gammaState(S[(i-1)*N+j-1]);
                c2 = D[(i-2)*N+j-1] + S[(i-1)*N+j] - gammaState(S[(i-2)*N+j-1]);
                c3 = D[(i-1)*N+j-2] + S[i*N+j-1] - gammaState(S[(i-1)*N+j-2]);
                c4 = D[(i-3)*N+j-1] + S[(i-2)*N+j] + S[(i-1)*N+j] - gammaState(S[(i-3)*N+j-1]);
                c5 = D[(i-1)*N+j-3] + S[i*N+j-2] + S[i*N+j-1] - gammaState(S[(i-1)*N+j-3]);
                D[i*N+j] = sixMax(0, c1, c2, c3, c4, c5);
            }
            if (D[i*N+j] > maxD) {
                maxD = D[i*N+j];
            }
        }
    }
    return maxD;
}