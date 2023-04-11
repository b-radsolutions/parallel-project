
#include <cstdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

void     normalize(double *src, double *dst, size_t n);
void     projection(double *vector, double *base, double *result, size_t n);
void     subtract(double *a, double *b, double *dst, size_t n);
void     cudaSetup();
void     cudaCleanup();
void     cleanupMatrix(double **A, size_t m);
double **createTestMatrix(size_t n);
double **allocateMatrix(size_t n);

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

int main() {
    cudaSetup();
    printf("cudaSetup() complete\n");

    // Create the matrix to use.
    srand(0);
    const size_t n = 10;
    double     **A, **Q;
    A = createTestMatrix(n);
    printf("Matrix A (%dx%d) successfully generated\n", n, n);
    Q = allocateMatrix(n);
    printf("Matrix Q (%dx%d) successfully initialized\n", n, n);

    // Run the procedure
    modified_gram_schmidt(A, n, n, Q);
    printf("Modified gram schmidt complete\n", n, n);

    // Cleanup
    cleanupMatrix(A, n);
    printf("cleanupMatrix(A) complete\n");
    cleanupMatrix(Q, n);
    printf("cleanupMatrix(Q) complete\n");
    cudaCleanup();
    printf("cudaCleanup() complete\n");
}
