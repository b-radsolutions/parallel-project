
#include "matrix-writer.hpp"
#include <fstream>
#include <iostream>
#include <string>

// A is the nxn matrix
int write_matrix_to_file_serial(double **A, size_t n, const std::string &filename) {
    // Open the file
    std::ofstream wf(filename.c_str(), std::ios::out | std::ios::binary);
    if (!wf) {
        std::cout << "Failed to open file " << filename << "\n";
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

int write_partial_matrix_to_file_serial(double **A, size_t m, size_t n,
                                        const std::string &filename) {
    // Open the file
    std::ofstream wf(filename.c_str(), std::ios::out | std::ios::binary);
    if (!wf) {
        std::cout << "Failed to open file " << filename << "\n";
        return 1;
    }

    // The first value we insert will be the size of the matrix as a size_t
    wf.write((char *)&n, sizeof(size_t));

    // Insert the rest of the values.
    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            wf.write((char *)&(A[x][y]), sizeof(double));
        }
    }

    wf.close();
    return 0;
}
