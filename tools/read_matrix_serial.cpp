
#include <assert.h>
#include <fstream>
#include <iostream>

double **read_matrix(const std::string& filename, size_t n) {
    // Open the file
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << "\n";
        return NULL;
    }

    // Calculate file size
    file.seekg(0, std::ios::end);
    size_t file_size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);

    // Get the size of the matrix from the file
    size_t read_n;
    file.read((char *)(&read_n), sizeof(size_t));
    if (read_n != n) {
        std::cerr << "Matrix is of size " << read_n << " but expected N= " << n <<"\n";
        return NULL;

    }
    // Make sure this file is properly sized
    size_t expected_size = sizeof(size_t) + sizeof(double) * n * n;
    if (file_size != expected_size) {
        std::cerr << "Matrix file seems to be an unexpected size. expected = "<< expected_size <<" ; got = "<< file_size << "\n";
        return NULL;

    }

    // Allocate the matrix
    double **A;
    A = (double **)malloc(sizeof(double *) * n);
    for (int x = 0; x < n; x++) {
        double *current = (double *)malloc(sizeof(double) * n);
        for (int y = 0; y < n; y++) {
            if (file.peek() == EOF) {
                std::cerr << "Reached End Of File too soon\n";
                return NULL;

            }
            double *loc = current + y;
            file.read((char *)(loc), sizeof(double));
        }
        A[x] = current;
    }

    file.close();
    return A;
}
