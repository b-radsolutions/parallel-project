#ifndef ORTHOGONALITY_TEST_HPP
#define ORTHOGONALITY_TEST_HPP

#include <stddef.h>

double orthoError(size_t m, size_t n, double **Q);
double frobeniusNorm(size_t m, size_t n, double **E);
double infNorm(size_t m, size_t n, double **E);
double oneNorm(size_t m, size_t n, double **E);

#endif
