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

#ifndef BRAD
#include "clockcycle.h"
#else
int clock_now(){
    return 0;
}
#endif

#define MASTER 0
#define clock_frequency 512000000.0

#define NUM_SIZES 8
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

    int         sizes[8] = {4, 16, 32, 64, 128, 256, 512, 1024};
    std::string types[4] = {"dense", "sparse", "well-conditioned", "ill-conditioned"};
//    if (world_rank == MASTER) {
//        printf("MPI Rank 0: ------Running Serial------\n");
//        for (int i = 0; i < NUM_SIZES; i++) {
//            int n = sizes[i];
//            for (int j = 0; j < NUM_MATRIX_VARAINTS; j++) {
//                string in_filename = "data/" + types[j] + "/" + to_string(n) + ".mtx";
//                cout << "GOING TO Matrix " << in_filename << "\t" << n << " by " << n
//                     << "in Serial\n ";
//                start = clock_now();
//                double **input_matrix = read_matrix(in_filename, n);
//                end = clock_now();
//                cout << "READ Matrix " << n << " by " << n << " in " << (end - start)
//                     << " cycles (" << (end - start) / clock_frequency << " secs)\n";
//
//                // Need to move the matrix to the device...
//                double **A = matrixHostToDevice(input_matrix, n, n);
//
//                cout << "RUNNING Serial Modified Gram-Schmidt\t" << types[j] << "\t" << n
//                     << "\n";
//
//                // Preallocate output
//                double **Q = allocateMatrix(n);
//
//                start = clock_now();
//                serial_modified_gram_schmidt(A, n, n, Q);
//                end = clock_now();
//
//                cout << "DONE in " << (end - start) << " cycles ("
//                     << (end - start) / clock_frequency << " secs)\n";
//
//                // Matrix Q is a device matrix and needs to be moved
//                double **deviceQ = matrixDeviceToHost(Q, n, n);
//
//                start = clock_now();
//                string out_filename = "out/ModifiedSerial" + to_string(n) + "by" +
//                                      to_string(n) + types[j] + ".mtx";
//                write_matrix_to_file_serial(deviceQ, n, out_filename);
//                end = clock_now();
//                cout << "WROTE TO FILE in " << (end - start) << "cycles("
//                     << (end - start) / clock_frequency << " secs)\n ";
//
//                // Delete host copy of Q.
//                for (size_t i = 0; i < n; i++)
//                    free(deviceQ[i]);
//                free(deviceQ);
//
//                cout << "RUNNING Serial Classic Gram-Schmidt\t" << types[j] << "\t" << n
//                     << "\n";
//
//                start = clock_now();
//                normal_gram_schmidt(A, n, n, Q);
//                end = clock_now();
//
//                cout << "DONE in " << (end - start) << " cycles ("
//                     << (end - start) / clock_frequency << " secs)\n";
//
//                // Matrix Q is (still) a device matrix and needs to be moved
//                deviceQ = matrixDeviceToHost(Q, n, n);
//
//                start = clock_now();
//                out_filename = "out/ClassicSerial" + to_string(n) + "by" + to_string(n) +
//                               types[j] + ".mtx";
//                write_matrix_to_file_serial(deviceQ, n, out_filename);
//                end = clock_now();
//                cout << "WROTE TO FILE in " << (end - start) << " cycles ("
//                     << (end - start) / clock_frequency << " secs)\n\n";
//
//                // Delete host copy of Q for classic GS.
//                for (size_t i = 0; i < n; i++)
//                    free(deviceQ[i]);
//                free(deviceQ);
//
//                // Delete A
//                cleanupMatrix(A, n);
//
//                // todo:: delete `input_matrix`
//            }
//            cout << "\n";
//        }
//    }

    // MPI portion

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == MASTER)
        printf("\n\n------Running Parallel------\n\n");

    for (int i = 0; i < 2/*NUM_SIZES*/; i++) {
        size_t n = sizes[i];
        int    rows_in = n / world_size;
        int    first_row = rows_in * world_rank;

        size_t m = rows_in;

        for (int j = 0; j < 1/*NUM_MATRIX_VARAINTS*/; j++) {
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

            // TODO THINGS TO TIME
            start = clock_now();
            string out_filename = "out/ModifiedParallel" + to_string(n) + "by" +
                                  to_string(n) + types[j] + ".mtx";
            write_partial_matrix(deviceQ, n, first_row, m, in_filename);
            end = clock_now();

//            for (size_t i = 0; i < n; i++)
//                free(deviceQ[i]);
//            free(deviceQ);

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
            out_filename = "out/ClassicParallel" + to_string(n) + "by" + to_string(n) +
                           types[j] + ".mtx";
            write_partial_matrix(deviceQ, n, first_row, m, in_filename);
            end = clock_now();
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