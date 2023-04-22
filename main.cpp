#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <string>


// #include "clockcycle.h"

#define MASTER 0
#define clock_frequency = 512000000.0;

long long unsigned clock_now() {
    return 0;
}

#define threads 1024

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


    int sizes[] = [ 4, 16, 32, 64, 128, 256, 512, 1024 ];
    std::string types[4] = [ 'Dense', 'Sparse', 'Well-conditioned', 'Ill-conditioned' ];
    if (world_rank == MASTER) {
        printf("MPI Rank 0: ------Running Serial------\n");
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                printf("Running Serial Modified Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
                start = clock_now();
                // TODO THINGS TO TIME
                end = clock_now();
                printf("MPI Rank 0: Serial Modified Gram-Schmidt Done in %llu cycles (%f secs)\n", (end - start),
                       (end - start) / clock_frequency);

                printf("Running Serial Classic Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
                start = clock_now();
                // TODO THINGS TO TIME
                end = clock_now();
                printf("MPI Rank 0: Serial Classic Gram-Schmidt Done in %d cycles (%f secs)\n", (end - start),
                       (end - start) / clock_frequency);
            }
        }
    }

  printf("------Running Parallel------\n");
    for (int i = 0; i < 8; i++){
        for (int j = 0; j < 4; j++){
            printf("Running Parallel Modified Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
            start = clock_now();
            // TODO THINGS TO TIME
            end = clock_now();
            printf("MPI Rank 0: Parallel Modified Gram-Schmidt Done in %llu cycles (%f secs)\n", (end - start),
                   (end - start) / clock_frequency);

            printf("Running Parallel Classic Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
            start = clock_now();
            // TODO THINGS TO TIME
            end = clock_now();

            printf("MPI Rank 0: Parallel Classic Gram-Schmidt Done in %llu cycles (%f secs)\n", (end - start),
                   (end - start) / clock_frequency);
        }
    }
    // Free memory
    MPI_Finalize();

    return 0;
}