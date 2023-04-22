#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string>
#include "clockcycle.h"

#define threads 1024

// Takes in args, and runs MPI
int main(int argc, char *argv[])
{
  // TODO: Read matrix with parallel IO
  double clock_frequency = 512000000;
  int myrank;
  int numranks;
  long long unsigned start = 0;
  long long unsigned end = 0;
  float time_in_secs = 0;
  int sizes[8] = [ 4, 16, 32, 64, 128, 256, 512, 1024 ];
  std::string types[4] = [ 'Dense', 'Sparse', 'Well-conditioned', 'Ill-conditioned' ];

  printf("------Running Serial------\n");
  for (int i = 0; i < 8; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      printf("Running Serial Modified Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
      start = clock_now();
      // THINGS TO TIME
      end = clock_now();
      time_in_secs = ((double)(end - start)) / clock_frequency;
      printf("%f\n", time_in_secs);

      printf("Running Serial Classic Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
      start = clock_now();
      // THINGS TO TIME
      end = clock_now();
      time_in_secs = ((double)(end - start)) / clock_frequency;
      printf("%f\n", time_in_secs);
    }
  }

  printf("------Running Parallel------\n");
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  for (int i = 0; i < 8; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      printf("Running Parallel Modified Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
      start = clock_now();
      // THINGS TO TIME
      end = clock_now();
      if (myrank == 0)
      {
        time_in_secs = ((double)(end - start)) / clock_frequency;
        printf("%f\n", time_in_secs);
      }
      printf("Running Parallel Classic Gram-Schmidt\t,%s\t%d\n", types[j], sizes[i]);
      start = clock_now();
      // THINGS TO TIME
      end = clock_now();
      if (myrank == 0)
      {
        time_in_secs = ((double)(end - start)) / clock_frequency;
        printf("%f\n", time_in_secs);
      }
    }
  }
  // Free memory
  MPI_Finalize();

  return 0;
}