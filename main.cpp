#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

  print("Running Serial Modified Gram-Schmidt\n");
  start = clock_now();
  // THINGS TO TIME
  end = clock_now();
  time_in_secs = ((double)(end - start)) / clock_frequency;
  printf("%f\n", time_in_secs);

  print("Running Serial Normal Gram-Schmidt\n");
  start = clock_now();
  // THINGS TO TIME
  end = clock_now();
  time_in_secs = ((double)(end - start)) / clock_frequency;
  printf("%f\n", time_in_secs);

  // PARALLELIZED
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);

  print("Running Parallel Modified Gram-Schmidt\n");
  start = clock_now();
  // THINGS TO TIME
  end = clock_now();
  if (myrank == 0)
  {
    time_in_secs = ((double)(end - start)) / clock_frequency;
    printf("%f\n", time_in_secs);
  }
  print("Running Parallel Normal Gram-Schmidt\n");
  start = clock_now();
  // THINGS TO TIME
  end = clock_now();
  if (myrank == 0)
  {
    time_in_secs = ((double)(end - start)) / clock_frequency;
    printf("%f\n", time_in_secs);
  }
  // Free memory
  MPI_Finalize();

  return 0;
}