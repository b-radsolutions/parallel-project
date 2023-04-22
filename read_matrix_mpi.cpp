
#include <mpi.h>
#include <stdio.h>

#define NUMBER_DIGITS_AFTER_POINT 16

/**
 * The matrix files should have the following format:
 *   Every file then has N*N entries. Every entry represents a number between 0 and 1.
 * Every entry has a leading 0, a decimal point, 8 numbers after, and then a space or a
 * newline character before the next entry.
 */

// '1.' + encoded digits + (' ' or '\n')
const size_t bytes_per_entry = 2 + NUMBER_DIGITS_AFTER_POINT + 1;

// Reads in the part of a matrix described
double **read_partial_matrix(size_t n, size_t first_row, size_t num_rows, MPI_File fp) {
    MPI_Offset offset = bytes_per_entry * first_row * n;
    MPI_Status status;
    size_t     number_to_read = num_rows * n;
    int     read_count;
    double    *buffer = (double *)malloc(sizeof(double) * number_to_read);

    MPI_File_read_at(fp, offset, buffer, num_rows, MPI_DOUBLE, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &read_count);

    printf("Read %d double values (starting at %ld, wanted to read %ld)\n", read_count,
           (size_t)offset, number_to_read);
}
