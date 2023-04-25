#pragma once
#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <stddef.h>

extern "C" {
void     normalize(double *src, double *dst, size_t vector_size);
void     distributedNormalize(double *src, double *dst, size_t partial_vector_size);
void     projection(double *vector, double *base, double *result, size_t vector_size);
void     distributedProjection(double *vector, double *base, double *result,
                               size_t partial_vector_size);
void     subtract(double *a, double *b, double *dst, size_t vector_size);
void     cudaSetup(size_t);
void     cudaCleanup();
void     cleanupMatrix(double **A, size_t number_vectors);
void     cleanupVector(double *vec);
double **createTestMatrix(size_t number_vectors, size_t vector_size);
double **allocateMatrix(size_t number_vectors, size_t vector_size);
double  *allocateVector(size_t vector_size);
double **matrixDeviceToHost(double **A, size_t number_vectors, size_t vector_size);
double **matrixHostToDevice(double **A, size_t number_vectors, size_t vector_size);
void     matrixCopy(double **A, double **B, size_t number_vectors, size_t vector_size);
double   dot(double *a, double *b, size_t vector_size);
double   serial_dot(double *a, double *b, size_t vector_size);
void     performModifiedGramSchmidtReduction(double **A, size_t number_vectors,
                                             size_t partial_vector_size,
                                             size_t completed_index);
}

#endif
