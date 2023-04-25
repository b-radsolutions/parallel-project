
#include <stdlib.h>

#include "matrix-operations.hpp"

void normal_gram_schmidt(double **A, size_t m, size_t n, double **Q) {
    // Copy over A into the output Q
    matrixCopy(A, Q, m, n);

    // Our first vector is already done
    normalize(Q[0], Q[0], n);

    // Create a temporary vector
    double *tmp = allocateVector(n);

    // For every vector after the first...
    for (size_t i = 1; i < m; i++) {
        // Subtract the projection of every vector that comes before
        for (size_t j = 0; j < i; j++) {
            projection(Q[i], Q[j], tmp, n);
            subtract(Q[i], tmp, Q[i], n);
        }
        // Finally, normalize that vector.
        normalize(Q[i], Q[i], n);
    }
    cleanupVector(tmp);
}

void parallel_gram_schmidt(double **A, size_t number_vectors, size_t vector_size,
                           double **Q) {
    // Copy over A into the output Q
    matrixCopy(A, Q, number_vectors, vector_size);

    // Our first vector is already done
    distributedNormalize(Q[0], Q[0], vector_size);

    // Create a temporary vector
    double *tmp = allocateVector(vector_size);

    // For every vector after the first...
    for (size_t i = 1; i < number_vectors; i++) {
        // Subtract the projection of every vector that comes before
        for (size_t j = 0; j < i; j++) {
            distributedProjection(Q[i], Q[j], tmp, vector_size);
            subtract(Q[i], tmp, Q[i], vector_size);
        }
        // Finally, normalize that vector.
        distributedNormalize(Q[i], Q[i], vector_size);
    }
    cleanupVector(tmp);
}

void serial_modified_gram_schmidt(double **A, size_t number_vectors, size_t vector_size,
                                  double **Q) {
    // Copy over A into the output Q
    matrixCopy(A, Q, number_vectors, vector_size);

    // Our first vector is already done
    normalize(Q[0], Q[0], vector_size);

    // I think this line is wrong; should be done on GPU
    double *tmp = allocateVector(vector_size);

    for (size_t i = 0; i < number_vectors - 1; i++) {
        // Subtract the projection of the previously-completed vector from the remaining
        for (size_t j = i + 1; j < number_vectors; j++) {
            projection(Q[j], Q[i], tmp, vector_size);
            subtract(Q[j], tmp, Q[j], vector_size);
        }
        // Normalize the vector we just completed
        normalize(Q[i + 1], Q[i + 1], vector_size);
    }
    cleanupVector(tmp);
}

void parallel_modified_gram_schmidt(double **A, size_t number_vectors, size_t vector_size,
                                    double **Q) {
    // Copy over A into the output Q
    matrixCopy(A, Q, number_vectors, vector_size);

    // Our first vector is already done
    distributedNormalize(Q[0], Q[0], vector_size);

    for (size_t i = 0; i < number_vectors - 1; i++) {
        // Perform the 1-op projection/subtraction on remaining vectors
        performModifiedGramSchmidtReduction(A, number_vectors, vector_size, i);
        // Normalize the vector we just completed
        distributedNormalize(Q[i + 1], Q[i + 1], vector_size);
    }
}
