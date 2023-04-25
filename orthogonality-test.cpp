#include "matrix-operations.hpp"
#include <cmath>
#include <stddef.h>
#include <string.h>
#include <fstream>
#include <iostream>

double **allocateHostMatrix(size_t n) {
    double **ret = (double **)malloc(sizeof(double *) * n);
    for (size_t i = 0; i < n; i++)
        ret[i] = (double *)malloc(sizeof(double) * n);
    return ret;
}

// Takes a matrix Q
// Returns a matrix E
//       where E = (Q^T *  Q) - I
double **orthoError(size_t m, size_t n, double **Q) {
    double **E = allocateHostMatrix(m);
    double tmp;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = i; j < m; j++) {
            tmp = serial_dot(Q[i], Q[j], n);
            // printf("%f", tmp);
            E[i][j] = tmp;
            // Subtract the identity
            if (i == j) {
                E[i][j] -= 1;
            }
        }
        for (size_t j = 0; j < i; j++) {
            E[i][j] = E[j][i];
        }
    }
    return E;
}

// Test for orthogonality Frobenius norm
//  concats matrix rows then finds the
//  2-norm of the vector in R^(m*n)
double frobeniusNorm(size_t m, size_t n, double **E) {
    // double total = 0.0;
    // for (size_t i = 0; i < m; i++) {
    //     for (size_t j = 0; j < n; j++) {
    //         total += E[i][j] * E[i][j];
    //     }
    // }
    // return sqrt(total);
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
    double error = 0.0;
    double tmp;
    for (size_t i = 0; i < m; i++) {
        double max_dot = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                tmp = serial_dot(E[j], E[i], n);
                if (tmp > max_dot) {
                    max_dot = tmp;
                }
            }
        }
        error += max_dot;
    }
    return error;
}

// Testing suite to gauge how orthogonal a matrix A is.
// takes the 2-norm of the errors of each column with
// respect to every other columns and finds the 1-norm of
// those sums.
double oneNorm(size_t m, size_t n, double **E) {
    double error = 0.0;
    double tmp;
    for (size_t i = 0; i < m; i++) {
        double total = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                tmp = serial_dot(E[j], E[i], n);
                total += tmp * tmp;
            }
        }
        error += sqrt(total);
    }
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
