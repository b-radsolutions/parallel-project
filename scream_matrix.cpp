
#include <stdio.h>
#include <stdlib.h>

double **read_matrix(char *filename, size_t n);

// Don't use this...
void print_matrix(double **A, size_t n) {
    printf("A (%lux%lu) = \n", n, n);
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            printf("%.4f", A[x][y]);
            if (y < n - 1)
                printf("\t");
        }
        printf("\n");
    }
}
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s [matrix file] [N]\n", argv[0]);
        return 1;
    }
    size_t   n = atol(argv[2]);
    double **A = read_matrix(argv[1], n);
    if (A)
        print_matrix(A, n);
}