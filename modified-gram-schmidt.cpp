
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "orthogonality-test.hpp"
#include "matrix-operations.hpp"

// For an m x n matrix A (A[m][n] is the bottom right entry, A has m columns with n rows
// each), orthonormalize the matrix A and put the result in the pre-allocated Q.
void modified_gram_schmidt(double **A, size_t m, size_t n, double **Q) {
    // Copy over A into the output Q
    matrixCopy(A, Q, m, n);

    // Our first vector is already done
    normalize(Q[0], Q[0], n);

    // I think this line is wrong; should be done on GPU
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
    free(tmp);
}

int main() {
    cudaSetup();
    printf("cudaSetup() complete\n");

    // Create the matrix to use.
    srand(0);
    const size_t n = 10;
    double     **A, **Q, **E;
    A = createTestMatrix(n);
    printf("Matrix A (%dx%d) successfully generated\n", n, n);
    Q = allocateMatrix(n);
    printf("Matrix Q (%dx%d) successfully initialized\n", n, n);

    // Run the procedure
    modified_gram_schmidt(A, n, n, Q);
    printf("Modified gram schmidt complete\n", n, n);

    //Test Gram-Schmidt accuracy
    E = orthoError(n, n, Q);   

    //Frobenius Norm
    double frob = frobeniusNorm(n, n, E);
    printf("Frobenius norm = %f", frob);

    //a, b norms
    double inf_norm = infNorm(n, n, E);
    printf("inf norm = %f", inf_norm);
    double one_norm = oneNorm(n, n, E);
    printf("one norm = %f", one_norm);

    // Cleanup
    cleanupMatrix(A, n);
    printf("cleanupMatrix(A) complete\n");
    cleanupMatrix(Q, n);
    printf("cleanupMatrix(Q) complete\n");
    cleanupMatrix(E, n);
    printf("cleanupMatrix(E) complete\n");
    cudaCleanup();
    printf("cudaCleanup() complete\n");
}
