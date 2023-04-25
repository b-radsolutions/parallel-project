
#include "../matrix-operations.hpp"
#include <cmath>
#include <stddef.h>
#include <string.h>

double serial_dot(double *a, double *b, size_t vector_size) {
    double res = 0;
    for (size_t i = 0; i < vector_size; i++) {
        res += a[i] * b[i];
    }
    return res;
}

double **allocateHostMatrix(size_t n) {
    double **ret = (double **)malloc(sizeof(double *) * n);
    for (size_t i = 0; i < n; i++)
        ret[i] = (double *)malloc(sizeof(double) * n);
    return ret;
}

double orthoError(size_t m, size_t n, double **Q) {
    double   tmp, total = 0.;
    double **cudaQ = matrixHostToDevice(Q, m, n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = i; j < m; j++) {
            // tmp = serial_dot(Q[i], Q[j], n);
            tmp = dot(Q[i], Q[j], n);
            total += 2 * tmp * tmp;
        }
    }
    cleanupMatrix(cudaQ, n);
    return sqrt(total);
}

// Test for orthogonality Frobenius norm
//  concats matrix rows then finds the
//  2-norm of the vector in R^(m*n)
double frobeniusNorm(size_t m, size_t n, double **E) {
    double total = 0.0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            total += E[i][j] * E[i][j];
        }
    }
    return sqrt(total);
}

// double** condition_number(size_t m, size_t n, double **A){
// }

// Testing suite to gauge how orthogonal a matrix A is.
// takes the infinity norm of the errors of each column in Q
// and then takes the 1-norm of each vector inf-norm
double infNorm(size_t m, size_t n, double **E) {
    double   error = 0.0;
    double   tmp;
    double **cudaQ = matrixHostToDevice(E, m, n);
    for (size_t i = 0; i < m; i++) {
        double max_dot = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                tmp = dot(cudaQ[j], cudaQ[i], n);
                // tmp = serial_dot(E[j], E[i], n);
                if (tmp > max_dot) {
                    max_dot = tmp;
                }
            }
        }
        error += max_dot;
    }
    cleanupMatrix(cudaQ, n);
    return error;
}

// Testing suite to gauge how orthogonal a matrix A is.
// takes the 2-norm of the errors of each column with
// respect to every other columns and finds the 1-norm of
// those sums.
double oneNorm(size_t m, size_t n, double **E) {
    double   error = 0.0;
    double   tmp;
    double **cudaQ = matrixHostToDevice(E, m, n);
    for (size_t i = 0; i < m; i++) {
        double total = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                tmp = dot(cudaQ[j], cudaQ[i], n);
                // tmp = serial_dot(E[j], E[i], n);
                total += tmp * tmp;
            }
        }
        error += sqrt(total);
    }
    cleanupMatrix(cudaQ, n);
    return error;
}

// //Testing suite to gauge how orthogonal a matrix A is.
// //
// double* orthogonality_test1(size_t m, size_t n, double **Q,  double *dot_totals, double
// *dot_maxes){
//     double *tmp;
//     for(size_t i = 0; i < m; i++) {
//         double total = 0.0
//         double max_dot = 0.0;
//         for(size_t j = 0; j < m; j++) {
//             if(i != j) {
//                serial_dot(Q[j], Q[i], tmp, n);
//                 total += *tmp;
//                 if(tmp > max_dot){
//                     max_dot = *tmp;
//                 }
//             }
//         }
//         dot_totals[i] = total;
//         dot_maxes[i] = max_dot;
//     }
//     free(tmp);
// }
