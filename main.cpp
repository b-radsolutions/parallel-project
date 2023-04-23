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
//#include "modified-gram-schmidt.cpp"


// #include "clockcycle.h"

#define MASTER 0
#define clock_frequency 512000000.0

long long unsigned clock_now() {
    return 0;
}

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


    int sizes[8] = {4, 16, 32, 64, 128, 256, 512, 1024 };
    std::string types[4] = { "dense", "sparse", "well-conditioned", "ill-conditioned" };
    /*
    if (world_rank == MASTER) {
        printf("MPI Rank 0: ------Running Serial------\n");
        for (int i = 0; i < 8; i++) {
            int n = sizes[i];
            for (int j = 0; j < 4; j++) {
                string in_filename = "data/" + types[j] + "/" + to_string(n) + ".mtx";
                cout << "READING Matrix " << in_filename << "\t" << n << " by " << n << " in Serial\n";
                start = clock_now();
                double **input_matrix = read_matrix(in_filename, n);
                end = clock_now();
                cout << "READ Matrix " << n << " by " << n << " in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n";

                cout << "RUNNING Serial Modified Gram-Schmidt\t" << types[j] << "\t" << sizes[i] << "\n";
                start = clock_now();

                double **output_matrix1 = input_matrix;
                end = clock_now();
                cout << "DONE in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n";
                // TODO THINGS TO TIME
                start = clock_now();
                string out_filename = "out/ModifiedSerial" + to_string(n) + "by" + to_string(n) + types[j] + ".mtx";
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
                out_filename = "out/ClassicSerial" + to_string(n) + "by" + to_string(n) + types[j] + ".mtx";
                write_matrix_to_file_serial(output_matrix2, n, out_filename);
                end = clock_now();
                cout << "WROTE TO FILE in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n\n";

            }
            cout << "\n";
        }
    }
    */
    MPI_Barrier( MPI_COMM_WORLD );
    printf("\n\n------Running Parallel------\n\n");
    for (int i = 0; i < 8; i++) {
        int n = sizes[i];
        int rows_in = n / world_size;
        int first_row = rows_in * world_rank;

        for (int j = 0; j < 4; j++) {
            string in_filename = "data/" + types[j] + "/" + to_string(n) + ".mtx";
            if (world_rank == MASTER) {
                cout << "MPI Rank " << world_rank << ": READING Matrix " << in_filename << "\t" << n << " by " << n
                     << " in Parallel\n";
            }
            start = clock_now();
            double **input_matrix = read_partial_matrix(n, first_row, rows_in, in_filename);
            end = clock_now();
            cout << "READ Matrix " << n << " by " << n << " in "<< (end - start) << " cycles ("<< (end - start) / clock_frequency << " secs)\n";

            if (world_rank == MASTER){
                cout << "RUNNING Parallel Modified Gram-Schmidt\t" << types[j] << "\t" << sizes[i] << "\n";
            }
            start = clock_now();

            double **output_matrix1 = input_matrix;
            end = clock_now();
            if (world_rank == MASTER) {
                cout << "DONE in " << (end - start) << " cycles (" << (end - start) / clock_frequency << " secs)\n";
            }
            // TODO THINGS TO TIME
            start = clock_now();
            string out_filename = "out/ModifiedParallel" + to_string(n) + "by" + to_string(n) + types[j] + ".mtx";
            //write_partial_matrix(output_matrix1, n, out_filename);
            end = clock_now();
            if (world_rank == MASTER) {
                cout << "WROTE TO FILE in " << (end - start) << " cycles (" << (end - start) / clock_frequency
                     << " secs)\n";
                cout << "RUNNING Parallel Classic Gram-Schmidt\t" << types[j] << "\t" << sizes[i] << "\n";
            }
            start = clock_now();
            // TODO THINGS TO TIME
            double **output_matrix2 = input_matrix;
            end = clock_now();
            if (world_rank == MASTER) {
                cout << "DONE in " << (end - start) << " cycles (" << (end - start) / clock_frequency << " secs)\n";
            }
            start = clock_now();
            out_filename = "out/ClassicParallel" + to_string(n) + "by" + to_string(n) + types[j] + ".mtx";
            //write_partial_matrix(output_matrix2, n, out_filename);
            end = clock_now();
            if (world_rank == MASTER) {
                cout << "WROTE TO FILE in " << (end - start) << " cycles (" << (end - start) / clock_frequency
                     << " secs)\n\n";
            }
        }
        cout << "\n";
    }

    // Free memory
    MPI_Finalize();

    return 0;
}