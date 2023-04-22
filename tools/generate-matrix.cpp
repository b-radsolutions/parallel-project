
#include "matrix-writer.hpp"
#include <fstream>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s [name of output file] [matrix size 'N']\n", argv[0]);
        return 1;
    }
    // Get the filename to write to
    char *filename = argv[1];
    // Get the number N of the matrix
    size_t n = std::atoi(argv[2]);

    printf("Generating matrix '%s' with size %lux%lu\n", filename, n, n);

    // Create and insert double values into matrix
    size_t   number_elements = n * n;
    double **A = (double **)malloc(sizeof(double *) * n);
    for (int x = 0; x < n; x++) {
        double *current = (double *)malloc(sizeof(double) * n);
        for (int y = 0; y < n; y++) {
            double value = ((double)rand() / (double)RAND_MAX);
            current[y] = value;
        }
        A[x] = current;
    }
    printf("Generated %lu elements.\n", number_elements);

    // Save the created matrix to file
    write_matrix_to_file(A, n, filename);
    printf("Wrote matrix to file.\n");
}
