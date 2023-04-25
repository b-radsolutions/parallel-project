
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "read_matrix_serial.hpp"

// Don't use this...
void print_matrix(double **A, size_t m, size_t n) {
    printf("A (%lux%lu) = \n", n, n);
    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            printf("%.16f", A[x][y]);
            if (y < n - 1)
                printf("\t");
        }
        printf("\n");
    }
}
int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        printf("Usage: %s [matrix file] (M) [N]\n", argv[0]);
        return 1;
    }
    double **A;
    size_t   m, n;
    if (argc == 4) {
        m = atol(argv[2]);
        n = atol(argv[3]);
        A = read_partial_matrix(argv[1], m, n);
    } else {
        n = atol(argv[2]);
        m = n;
        A = read_matrix(argv[1], n);
    }
    if (A)
        print_matrix(A, m, n);
}