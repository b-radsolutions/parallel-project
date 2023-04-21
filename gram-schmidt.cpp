
#include <stdlib.h>

#include "matrix-operations.hpp"

void normal_gram_schmidt(double **A, size_t m, size_t n, double **Q) {
    // Copy over A into the output Q
    matrixCopy(A, Q, m, n);

    // Our first vector is already done
    normalize(Q[0], Q[0], n);

    // Create a temporary vector
    double *tmp; // todo:: allocate tmp

    // For every vector after the first...
    for (size_t i = 1; i < m; i++) {
        // Subtract the projection of every vector that comes before
        for (size_t j = 0; j < i; j++) {
            projection(Q[i], Q[j], tmp, n);
            subtract(Q[i], tmp, Q[i], n);
        }
        // Finally, normalize that vector.
        normalize(Q[i + 1], Q[i + 1], n);
    }

    // todo:: cleanup tmp
}