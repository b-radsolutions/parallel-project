
#include "matrix-writer.hpp"
#include <fstream>

// A is the nxn matrix
int write_matrix_to_file(double **A, size_t n, char *filename) {
    // Open the file
    std::ofstream wf(filename, std::ios::out | std::ios::binary);
    if (!wf) {
        printf("Failed to open file '%s'\n", filename);
        return 1;
    }

    // The first value we insert will be the size of the matrix as a size_t
    wf.write((char *)&n, sizeof(size_t));

    // Insert the rest of the values.
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            wf.write((char *)&(A[x][y]), sizeof(double));
        }
    }

    wf.close();
    return 0;
}
