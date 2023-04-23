#pragma once
#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

void     normalize(double *src, double *dst, size_t n);
void     projection(double *vector, double *base, double *result, size_t n);
void     subtract(double *a, double *b, double *dst, size_t n);
void     cudaSetup();
void     cudaCleanup();
void     cleanupMatrix(double **A, size_t m);
double **createTestMatrix(size_t n);
double **allocateMatrix(size_t n);
void     matrixCopy(double **A, double **B, size_t m, size_t n);
double   dot(double *a, double *b, size_t n);
void     performModifiedGramSchmidtReduction(double **A, size_t m, size_t n,
                                             size_t completed_index);

#endif
