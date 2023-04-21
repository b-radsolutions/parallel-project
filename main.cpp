#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "clockcycle.h"


#define threads 1024
// Takes in args, and runs MPI
int main(int argc, char *argv[])
{
  //Read matrix with parallel IO
  double clock_frequency = 512000000;
  int myrank;
  int numranks;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
 // SWITCH FOR EACH CASE OF SERIAL/Parallel Gram-Schmidt
  long long unsigned start = clock_now();
  //THINGS TO TIME
  long long unsigned end = clock_now();
  // Print timing & result
  if (myrank == 0)
  {
    float time_in_secs = ((double)(end - start)) / clock_frequency;
    printf("%lld %f\n", result1, time_in_secs);
  }
  // Free memory
  

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}