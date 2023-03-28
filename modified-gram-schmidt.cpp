
#include <cstdlib.h>
#include <stddef.h>
#include <string.h>

void normalize(double *src, double *dst, size_t n);
void projection(double *vector, double *base, double *result, size_t n);
void subtract(double *a, double *b, double *dst, size_t n);

// For an m x n matrix A (A[m][n] is the bottom right entry, A has m columns with n rows
// each), orthonormalize the matrix A and put the result in the pre-allocated Q.
void modified_gram_schmidt(double **A, size_t m, size_t n, double **Q) {
    // Our first vector is already done
    normalize(Q[0], Q[0], n);

    // Copy over the rest of A into the output Q
    for (size_t i = 1; i < m; i++) {
        memcpy(Q[i], A[i], sizeof(double) * n);
    }

    double *tmp = (double *)malloc(sizeof(double) * n);
    for (size_t i = 0; i < m - 1; i++) {
        // Subtract the projection of the previously-completed vector from the remaining
        for (size_t j = i + 1; j < m; j++) {
            projection(Q[j], Q[i], tmp, n);
            subtract(Q[j], tmp, Q[j], n);
        }
        // Normalize the vector we just completed
        normalize(Q[i + 1], Q[i + 1], n);
    }
}
