#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix-operations.hpp"
#include "orthogonality-test.hpp"
#define M 8
#define FEPSILON 0.0001

double **matrixA;     // 1, 2, 3, ... n-1, n; 1, 2, 3, ... n-1, n; 1, 2, 3, ... n-1, n ...
double **matrixB;     // n, n-1, ... 3, 2, 1; n, n-1, ... 3, 2, 1; n, n-1, ... 3, 2, 1 ...
double **matrixI;     // n, n-1, ... 3, 2, 1; n, n-1, ... 3, 2, 1; n, n-1, ... 3, 2, 1 ...
double **matrixZeros; // 0, 0, 0, ... 0; 0, 0, 0, ... 0; 0, 0, 0, ... 0 ...
double **matrixOnes;   // 1, 1, 1, ... 1; 1, 1, 1, ... 1; 1, 1, 1, ... 1 ...
double **E;

void createTestStructures() {
    matrixA = allocateMatrix(M);
    matrixB = allocateMatrix(M);
    matrixI = allocateMatrix(M);
    matrixZeros = allocateMatrix(M);
    matrixOnes = allocateMatrix(M);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            matrixA[j][i] = (double)(i + 1);
        }
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            matrixB[j][i] = (double)(M - i);
        }
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            if (i == j) {
                matrixI[j][i] = 1;
            }
            else {
                matrixI[j][i] = 0;
            }
        }
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            matrixZeros[j][i] = 0;
        }
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            matrixOnes[j][i] = 1;
        }
    }   
}


void testOrthoError(){
    double  expected;
    //for identity matrix
    E = orthoError(M, M, matrixI);
    printf("Οrthogonal Error of the identity matrix\n");
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            expected = 0;
            if (fabs(expected - E[i][j]) >= FEPSILON) {
            printf("Failed. orthoError for the identity matrix should be the zero matrix"
                   "%f ; got: %f\n",
                   expected, E[i][j]);
            exit(1);
            }
        }
    }

    //for ones matrix
   E = orthoError(M, M, matrixOnes);
    printf("Οrthogonal Error of the ones matrix\n");
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            if (i==j) {
                expected = M-1;
            }
            else {
                expected = M;
            }

            if (fabs(expected - E[i][j]) >= FEPSILON) {
            printf("Failed. orthoError for the ones matrix should have %d-1 on the main diagonal and %d everywhere else"
                   "%f ; got: %f\n", M, M,
                   expected, E[i][j]);
            exit(1);
            }
        }
    }
}

void testFrob() {
    double errorNorm, expected;
    printf("Frobenius Norm for the Identiy matrix\n");
    errorNorm = frobeniusNorm(M, M, matrixI);
    expected = M;
    if (fabs(expected - errorNorm) >= FEPSILON) {
        printf("Failed. Frobenius Norm for I is"
            "%f ; got: %f\n",
            expected, errorNorm);
        exit(1);
    }

    printf("Frobenius Norm for matrixA\n");
    errorNorm = frobeniusNorm(M, M, matrixA);
    expected = ((M*(M+1)*(2*M+1))/6) * M;
    if (fabs(expected - errorNorm) >= FEPSILON) {
        printf("Failed. Frobenius Norm for matrix A is "
            "%f ; got: %f\n",
            expected, errorNorm);
        exit(1);
    }
}


void testInfNorm() {
    double errorNorm, expected;
    printf("Inf Norm for the Identiy matrix\n");
    errorNorm = infNorm(M, M, matrixI);
    expected = M;
    if (fabs(expected - errorNorm) >= FEPSILON) {
        printf("Failed. Inf Norm for I is"
            "%f ; got: %f\n",
            expected, errorNorm);
        exit(1);
    }
}

void testOneNorm() {
    double errorNorm, expected;
    printf("One Norm for the Identiy matrix\n");
    errorNorm = oneNorm(M, M, matrixI);
    expected = M;
    if (fabs(expected - errorNorm) >= FEPSILON) {
        printf("Failed. Inf Norm for I is"
            "%f ; got: %f\n",
            expected, errorNorm);
        exit(1);
    }
}

void cleanUp() {
    cleanupMatrix(matrixA, M);
    cleanupMatrix(matrixB, M);
    cleanupMatrix(matrixI, M);
    cleanupMatrix(matrixZeros, M);
    cleanupMatrix(matrixOnes, M);
}

int main() {
    createTestStructures();
    testOrthoError();
    testFrob();
    testInfNorm();
    testOneNorm();
    cleanUp();
}
