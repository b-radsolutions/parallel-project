
#include <mpi.h>
#include <stdio.h>
#include <string>



// Reads in the part of a matrix described
double ** read_partial_matrix(size_t n, size_t first_row, size_t num_rows, const std::string& filename) {
    //MPI_Offset offset = bytes_per_entry * first_row * n;
    MPI_Status status;
    MPI_File fp;
   // size_t     number_to_read = num_rows * n;
    int     read_count;

    int rc = MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
    if (rc != 0){
        std::cerr << "Failed to open file " << filename << "\n";
        return NULL;
    }

    double **A;
    A = (double **)malloc(sizeof(double *) * num_rows);
    MPI_File_seek(fp, sizeof(size_t) + (sizeof(double)*first_row*n), MPI_SEEK_SET);

    for (int i = 0; i < num_rows; ++i) {
        double *current = (double *)malloc(sizeof(double) * n);
        MPI_File_read(fp, current, n, MPI_DOUBLE, &status);
        A[i] = current;
    }
    //MPI_Get_count(&status, MPI_DOUBLE, &read_count);

    MPI_File_close( &fp );
    return A;
}
/*
int write_partial_matrix(double ** A, size_t n, size_t first_row, size_t num_rows, const std::string& filename){
    MPI_File fp;
    int rc = MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
    if (rc != 0){
        std::cerr << "Failed to open file " << filename << "\n";
        return 1;
    }
    MPI_Offset = 0;
    MPI_Status status;
    MPI_File_seek(fp, sizeof(size_t) + (sizeof(double)*first_row*n), MPI_SEEK_SET);
    for (int i = 0; i < num_rows; ++i) {
        MPI_File_write(fp, A[i], n, MPI_DOUBLE, MPI_Status *status)
    }
    return 0;
}
*/