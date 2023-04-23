#pragma once
#ifndef MPI_HELPER_HPP
#define MPI_HELPER_HPP

#include <stddef.h>

double *my_MPIReduce(double *partial_dots, size_t count, double *complete_dots);

#endif
