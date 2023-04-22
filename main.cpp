#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <string>
#include <cstring>

#include "tools/matrix-writer.cpp"
#include "tools/read_matrix_mpi.cpp"
#include "tools/read_matrix_serial.cpp"
#include "matrix-operations.hpp"


// #include "clockcycle.h"

#define MASTER 0
#define clock_frequency 512000000.0

long long unsigned clock_now() {
    return 0;
}

#define threads 1024
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


    int sizes[8] = { 16, 32, 64, 128, 256, 512, 1024 };
    std::string types[4] = { "Dense", "Sparse", "Well-conditioned", "Ill-conditioned" };
    if (world_rank == MASTER) {
        printf("MPI Rank 0: ------Running Serial------\n");
        for (int i = 0; i < 7; i++) {
            int n = sizes[i];

            cout << "READING Matrix " << n << " by " << n << " in Serial\n";
            string in_filename = "tools/gen/" + to_string(n) + ".mtx";
            start = clock_now();
            double **input_matrix = read_matrix(in_filename, n);
            end = clock_now();
            cout << "READ Matrix " << n << " by " << n << " in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n\n";

            for (int j = 0; j < 4; j++) {
                cout << "RUNNING Serial Modified Gram-Schmidt\t" << types[j] << "\t" << sizes[i] << "\n";
                start = clock_now();

                double **output_matrix1;
                end = clock_now();
                cout << "DONE in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n";

                start = clock_now();
                string out_filename = "out/ModifiedSerial" + to_string(n) + "by" + to_string(n) + types[j] + ".mts";
                write_matrix_to_file_serial(output_matrix1, n, out_filename);
                end = clock_now();
                cout << "WROTE TO FILE in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n";

                cout << "RUNNING Serial Classic Gram-Schmidt\t" << types[j] << "\t" << sizes[i] << "\n";
                start = clock_now();
                // TODO THINGS TO TIME
                double **output_matrix2 = input_matrix;
                end = clock_now();
                cout << "DONE in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n";

                start = clock_now();
                string out_filename = "out/ClassicSerial" + to_string(n) + "by" + to_string(n) + types[j] + ".mts";
                write_matrix_to_file_serial(output_matrix2, n, out_filename);
                end = clock_now();
                cout << "WROTE TO FILE in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n";

            }
            cout << "\n";
        }
    }
/*
  printf("------Running Parallel------\n");
    for (int i = 0; i < 8; i++){
        for (int j = 0; j < 4; j++){
            printf("Running Parallel Modified Gram-Schmidt\t,%s\t%d\n", &types[j], sizes[i]);
            start = clock_now();
            // TODO THINGS TO TIME
            end = clock_now();
            printf("MPI Rank 0: Parallel Modified Gram-Schmidt Done in %llu cycles (%f secs)\n", (end - start),
                   (end - start) / clock_frequency);

            printf("Running Parallel Classic Gram-Schmidt\t,%s\t%d\n", &types[j], sizes[i]);
            start = clock_now();
            // TODO THINGS TO TIME
            end = clock_now();

            printf("MPI Rank 0: Parallel Classic Gram-Schmidt Done in %llu cycles (%f secs)\n", (end - start), (end - start) / clock_frequency);
        }
    }
    */
    // Free memory
    MPI_Finalize();

    return 0;
}