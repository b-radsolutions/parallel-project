#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>

#include "gram-schmidt.hpp"
#include "matrix-operations.hpp"
#include "tools/matrix-writer.cpp"
#include "tools/read_matrix_mpi.cpp"
#include "tools/read_matrix_serial.cpp"

#include "clockcycle.h"

#define MASTER 0
#define clock_frequency 512000000.0

#define NUM_SIZES 11
#define NUM_MATRIX_VARAINTS 4

using namespace std;

// Takes in args, and runs MPI
int main(int argc, char *argv[]) {
    long long unsigned start = clock_now();
    long long unsigned end = clock_now();

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Cuda setup must happen after MPI is initialized so all MPI
    // ranks create the helper structures locally.
    cudaSetup(world_rank);

    int         sizes[11] = {4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    std::string types[4] = {"dense", "sparse", "well-conditioned", "ill-conditioned"};

    // MPI portion

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == MASTER)
        printf("\n\n------Running Parallel------\n\n");

    const int start_size_index = 3;
    for (int i = start_size_index; i < NUM_SIZES; i++) {
        size_t n = sizes[i];
        int    rows_in = n / world_size;
        int    first_row = rows_in * world_rank;

        size_t m = rows_in;

        for (int j = 0; j < NUM_MATRIX_VARAINTS; j++) {
            string in_filename = "data/" + types[j] + "/" + to_string(n) + ".mtx";
            if (world_rank == MASTER) {
                cout << "MPI Rank " << world_rank << ": READING Matrix " << in_filename
                     << "\t" << n << " by " << n << " in Parallel\n";
            }
            start = clock_now();
            double **input_matrix = read_partial_matrix(n, first_row, m, in_filename);
            end = clock_now();

            if (world_rank == MASTER)
                cout << "READ Matrix " << n << " by " << n << " in " << (end - start)
                     << " cycles (" << (end - start) / clock_frequency << " secs)\n";

            // Need to move this matrix into the device
            double **A = matrixHostToDevice(input_matrix, n, m);
            // Also need to initialize the result matrix Q
            double **Q = allocateMNMatrix(n, m);

            if (world_rank == MASTER) {
                cout << "RUNNING Parallel Modified Gram-Schmidt\t" << types[j] << "\t"
                     << sizes[i] << "\n";
            }

            start = clock_now();
            parallel_modified_gram_schmidt(A, m, n, Q);
            end = clock_now();

            if (world_rank == MASTER) {
                cout << "DONE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n";
            }

            // Move the matrix from the device back to the host
            double **deviceQ = matrixDeviceToHost(Q, n, m);

            start = clock_now();
            string out_filename = "out/ModifiedParallel" + to_string(n) + "by" +
                                  to_string(n) + types[j] + ".mtx";
            double **B;
            if (MASTER == world_rank) {
                B = (double **)malloc(sizeof(double *) * n);
            }
            for (int k = 0; k < n; ++k) {
                double *current = (double *)malloc(sizeof(double) * rows_in);
                for (int l = 0; l < rows_in; ++l) {
                    current[l] = deviceQ[l][k];
                }
                double *tmp;
                if (world_rank == MASTER) {
                    B[k] = (double *)malloc(sizeof(double) * n);
                    tmp = B[k];
                }
                MPI_Gather(current, rows_in, MPI_DOUBLE, tmp, rows_in, MPI_DOUBLE, MASTER,
                           MPI_COMM_WORLD);
                free(current);
            }
            if (MASTER == world_rank) {
                write_matrix_to_file_serial(B, n, out_filename);
                for (int k = 0; k < n; ++k) {
                    free(B[k]);
                }
                free(B);
            }

            end = clock_now();
            MPI_Barrier(MPI_COMM_WORLD);

            for (size_t i = 0; i < m; i++)
                free(deviceQ[i]);
            free(deviceQ);

            if (world_rank == MASTER) {
                cout << "WROTE TO FILE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n";
                cout << "RUNNING Parallel Classic Gram-Schmidt\t" << types[j] << "\t"
                     << sizes[i] << "\n";
            }

            start = clock_now();
            parallel_gram_schmidt(A, m, n, Q);
            end = clock_now();

            if (world_rank == MASTER) {
                cout << "DONE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n";
            }

            // Move the matrix from the device back to the host
            deviceQ = matrixDeviceToHost(Q, n, m);

            start = clock_now();

            // Every rank will save their own file.
            out_filename = "out/ClassicParallel" + to_string(n) + "by" + to_string(n) +
                           types[j] + "part" + to_string(world_rank) + ".mtx";
            write_partial_matrix_to_file_serial(deviceQ, rows_in, n, out_filename);

            // if (MASTER == world_rank) {
            //     B = (double **)malloc(sizeof(double *) * n);
            // }
            // for (int k = 0; k < n; ++k) {
            //     double *current = (double *)malloc(sizeof(double) * rows_in);
            //     for (int l = 0; l < rows_in; ++l) {
            //         current[l] = deviceQ[l][k];
            //     }
            //     double *tmp;
            //     if (world_rank == MASTER) {
            //         B[k] = (double *)malloc(sizeof(double) * n);
            //         tmp = B[k];
            //     }
            //     MPI_Gather(current, rows_in, MPI_DOUBLE, tmp, rows_in, MPI_DOUBLE,
            //     MASTER,
            //                MPI_COMM_WORLD);
            //     free(current);
            // }
            // if (MASTER == world_rank) {
            //     write_matrix_to_file_serial(B, n, out_filename);
            //     for (int k = 0; k < n; ++k) {
            //         free(B[k]);
            //     }
            //     free(B);
            // }

            end = clock_now();
            MPI_Barrier(MPI_COMM_WORLD);

            if (world_rank == MASTER) {
                cout << "WROTE TO FILE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n\n";
            }

            for (size_t i = 0; i < m; i++)
                free(deviceQ[i]);
            free(deviceQ);
        }
        if (world_rank == MASTER)
            cout << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Free memory
    MPI_Finalize();

    cudaCleanup();

    return 0;
}
