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

void write_matrix(int world_rank, size_t VECTOR_SIZE, size_t NUMBER_VECTORS,
                  double **deviceMatrix, string out_filename_prefix) {
    // Every rank will save their own file.
    // out_filename = "out/ClassicParallel" + to_string(n) + "by" + to_string(n) +
    //                 types[j] + "part" + to_string(world_rank) + ".mtx";

    string out_filename = out_filename_prefix + "_part_" + to_string(world_rank) + ".mtx";
    write_partial_matrix_to_file_serial(deviceMatrix, NUMBER_VECTORS, VECTOR_SIZE,
                                        out_filename);

    // double **B;
    // if (MASTER == world_rank) {
    //     B = (double **)malloc(sizeof(double *) * NUMBER_VECTORS);
    // }
    // for (int k = 0; k < NUMBER_VECTORS; ++k) {
    //     double *current = (double *)malloc(sizeof(double) * VECTOR_SIZE);
    //     for (int l = 0; l < NUMBER_VECTORS; ++l) {
    //         current[l] = deviceMatrix[l][k];
    //     }
    //     double *tmp;
    //     if (world_rank == MASTER) {
    //         B[k] = (double *)malloc(sizeof(double) * VECTOR_SIZE);
    //         tmp = B[k];
    //     }
    //     MPI_Gather(current, NUMBER_VECTORS, MPI_DOUBLE, tmp, NUMBER_VECTORS,
    //     MPI_DOUBLE,
    //                MASTER, MPI_COMM_WORLD);
    //     free(current);
    // }
    // if (MASTER == world_rank) {
    //     write_matrix_to_file_serial(B, NUMBER_VECTORS, out_filename);
    //     for (int k = 0; k < NUMBER_VECTORS; ++k) {
    //         free(B[k]);
    //     }
    //     free(B);
    // }
}

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

    string prefix = "out/" + to_string(world_size) + "/";

    // MPI portion

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == MASTER) {
        printf("WORLD SIZE: %d\n\n", world_size);
        printf("\n\n------Running Parallel------\n\n");
    }

    for (int i = 0; i < NUM_SIZES; i++) {
        // const int start_size_index = 3;
        // for (int i = start_size_index; i < start_size_index + 1; i++) {
        size_t NUMBER_VECTORS = sizes[i];
        size_t VECTOR_SIZE = NUMBER_VECTORS / world_size;
        int    first_row = VECTOR_SIZE * world_rank;

        for (int j = 0; j < NUM_MATRIX_VARAINTS; j++) {
            // for (int j = 0; j < 1; j++) {
            string in_filename =
                "data/" + types[j] + "/" + to_string(NUMBER_VECTORS) + ".mtx";
            if (world_rank == MASTER) {
                cout << "MPI Rank " << world_rank << ": READING Matrix " << in_filename
                     << "\t" << NUMBER_VECTORS << " by " << NUMBER_VECTORS
                     << " in Parallel\n";
            }
            start = clock_now();
            double **input_matrix =
                read_partial_matrix(VECTOR_SIZE, first_row, NUMBER_VECTORS, in_filename);
            end = clock_now();

            if (world_rank == MASTER)
                cout << "READ Matrix " << NUMBER_VECTORS << " by " << NUMBER_VECTORS
                     << " in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n";

            // Need to move this matrix into the device
            double **A = matrixHostToDevice(input_matrix, NUMBER_VECTORS, VECTOR_SIZE);
            // Also need to initialize the result matrix Q
            double **Q = allocateMatrix(NUMBER_VECTORS, VECTOR_SIZE);

            if (world_rank == MASTER) {
                cout << "RUNNING Parallel Modified Gram-Schmidt\t" << types[j] << "\t"
                     << sizes[i] << "\n";
            }

            start = clock_now();
            parallel_modified_gram_schmidt(A, NUMBER_VECTORS, VECTOR_SIZE, Q);
            end = clock_now();

            if (world_rank == MASTER) {
                cout << "DONE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n";
            }

            // Move the matrix from the device back to the host
            double **deviceQ = matrixDeviceToHost(Q, NUMBER_VECTORS, VECTOR_SIZE);

            // Write the matrix
            start = clock_now();
            string out_filename = prefix + "ModifiedParallel" +
                                  to_string(NUMBER_VECTORS) + "by" +
                                  to_string(NUMBER_VECTORS) + types[j];
            write_matrix(world_rank, VECTOR_SIZE, NUMBER_VECTORS, deviceQ, out_filename);
            end = clock_now();
            MPI_Barrier(MPI_COMM_WORLD);

            for (size_t i = 0; i < NUMBER_VECTORS; i++)
                free(deviceQ[i]);
            free(deviceQ);

            if (world_rank == MASTER) {
                cout << "WROTE TO FILE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n";
                cout << "RUNNING Parallel Classic Gram-Schmidt\t" << types[j] << "\t"
                     << sizes[i] << "\n";
            }

            A = matrixHostToDevice(input_matrix, NUMBER_VECTORS, VECTOR_SIZE);

            start = clock_now();
            parallel_gram_schmidt(A, NUMBER_VECTORS, VECTOR_SIZE, Q);
            end = clock_now();

            if (world_rank == MASTER) {
                cout << "DONE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n";
            }

            // Move the matrix from the device back to the host
            deviceQ = matrixDeviceToHost(Q, NUMBER_VECTORS, VECTOR_SIZE);

            start = clock_now();
            out_filename = prefix + "ClassicParallel" + to_string(NUMBER_VECTORS) + "by" +
                           to_string(NUMBER_VECTORS) + types[j];
            write_matrix(world_rank, VECTOR_SIZE, NUMBER_VECTORS, deviceQ, out_filename);
            end = clock_now();
            MPI_Barrier(MPI_COMM_WORLD);

            if (world_rank == MASTER) {
                cout << "WROTE TO FILE in " << (end - start) << " cycles ("
                     << (end - start) / clock_frequency << " secs)\n\n";
            }

            for (size_t i = 0; i < NUMBER_VECTORS; i++)
                free(deviceQ[i]);
            free(deviceQ);

            // todo:: cleanup
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
