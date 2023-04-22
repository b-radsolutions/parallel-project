
#include "matrix-writer.hpp"
#include <fstream>

#define sparity .1

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s [name of output file] [matrix size 'N'] [Type(1,2,3,4) '1']\n", argv[0]);
        return 1;
    }
    // Get the filename to write to
    char *filename = argv[1];
    // Get the number N of the matrix
    size_t n = std::atoi(argv[2]);

    // generate type of matrix
    // 1. Dense
    // 2. Sparse
    // 3. Well Conditioned
    // 4. Ill Conditioned
    size_t type = std::atoi(argv[3]);

    // Everyone uses
    size_t number_elements = n * n;
    double **A = (double **)malloc(sizeof(double *) * n);

    if (type == 1)
    {
        printf("Generating matrix '%s' with size %lux%lu\n", filename, n, n);

        // insert double values into matrix
        for (int x = 0; x < n; x++)
        {
            double *current = (double *)malloc(sizeof(double) * n);
            for (int y = 0; y < n; y++)
            {
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
    else if (type == 2)
    {
        printf("Generating matrix '%s' with size %lux%lu\n", filename, n, n);

        // insert double values into matrix
        for (int x = 0; x < n; x++)
        {
            double *current = (double *)malloc(sizeof(double) * n);
            for (int y = 0; y < n; y++)
            {
                if (x == y)
                {
                    current[y] = 1;
                }
                else
                {
                    current[y] = 0;
                }
            }
            A[x] = current;
        }
        int numNonZeros = sparity * number_elements; // Calculate the number of non-zero elements
        // Generate random non-zero elements
        for (int i = 0; i < numNonZeros; i++)
        {
            int x = rand() % n;
            int y = rand() % n;
            while (x == y)
            {
                x = rand() % n;
                y = rand() % n;
            }
            double value = ((double)rand() / (double)RAND_MAX);
            A[x][y] = value;
        }
        printf("Generated %lu elements.\n", number_elements);

        // Save the created matrix to file
        write_matrix_to_file(A, n, filename);
        printf("Wrote matrix to file.\n");
    }
    else if (type == 3)
    {
        printf("Generating matrix '%s' with size %lux%lu\n", filename, n, n);

        // insert double values into matrix
        for (int x = 0; x < n; x++)
        {
            double *current = (double *)malloc(sizeof(double) * n);
            for (int y = 0; y < n; y++)
            {
                current[y] = 0;
            }
            A[x] = current;
        }
        for (int x = 0; x < n; x++)
        {
            if (x < n / 2)
            {
                A[x][x] = (double)rand() / RAND_MAX * 0.99 + 0.01; // Small values
            }
            else
            {
                A[x][x] = (double)rand() / RAND_MAX * 99 + 1; // Large values
            }
        }

        // Generate a random non-diagonal matrix with small values
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < n; y++)
            {
                if (x != y)
                {
                    A[x][y] = (double)rand() / RAND_MAX * 0.99 + 0.01; // Small values
                }
            }
        }
        printf("Generated %lu elements.\n", number_elements);

        // Save the created matrix to file
        write_matrix_to_file(A, n, filename);
        printf("Wrote matrix to file.\n");
    }
    else if (type == 4)
    {
        printf("Generating matrix '%s' with size %lux%lu\n", filename, n, n);

        // Create and insert double values into matrix
        size_t number_elements = n * n;
        double **A = (double **)malloc(sizeof(double *) * n);
        for (int x = 0; x < n; x++)
        {
            double *current = (double *)malloc(sizeof(double) * n);
            for (int y = 0; y < n; y++)
            {
                current[y] = 0;
            }
            A[x] = current;
        }
        // Modify the matrix to make it ill-conditioned
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < n; y++)
            {
                if (x == y)
                {
                    A[x][y] += 1; // Add 1 to the diagonal entries
                }
                else
                {
                    A[x][y] -= 0.01; // Subtract a small constant from non-diagonal entries
                }
            }
        }
        printf("Generated %lu elements.\n", number_elements);

        // Save the created matrix to file
        write_matrix_to_file(A, n, filename);
        printf("Wrote matrix to file.\n");
    }
    else
    {
        printf("UNKNOWN TYPE\n");
    }
}
