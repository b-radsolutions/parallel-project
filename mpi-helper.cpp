
#include "mpi-helper.hpp"
#include <mpi.h>

// Given the partial dot products, MPI reduces
void my_MPIReduce(double *partial_dots, size_t count, double *complete_dots) {
    MPI_Allreduce(partial_dots, complete_dots, count, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
}
