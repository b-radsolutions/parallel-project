
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10
#define GRID_SIZE 16
#define FEPSILON 0.0001

__global__ void vector_subtraction(int n, double *x, double *y);
__global__ void vector_dot_product(int n, double *x, double *y, double *result);
__global__ void many_vector_subtractions(double *base, double **vectors, double *scalars,
                                         size_t num_vectors, size_t n);
__global__ void many_vector_dot_product(double *base, double **vectors,
                                        size_t num_vectors, size_t n, double *result);

const size_t arrSize = N * sizeof(double);

double *arrayA;     // 1, 2, 3, ... n-1, n
double *arrayB;     // 1, 2, 3, ... n-1, n
double *arrayC;     // n, n-1, ... 3, 2, 1
double *arrayEmpty; // 0, 0, 0, ... 0
double *arrayOne;   // 1, 1, 1, ... 1
double *device_result;
double *result;

void createTestStructures() {
    cudaMalloc(&arrayA, arrSize);
    cudaMalloc(&arrayB, arrSize);
    cudaMalloc(&arrayC, arrSize);
    cudaMalloc(&arrayEmpty, arrSize);
    cudaMalloc(&arrayOne, arrSize);
    cudaMalloc(&device_result, sizeof(double));
    result = (double *)malloc(sizeof(double));

    double *tmp = (double *)malloc(arrSize);
    for (size_t i = 0; i < N; i++) {
        tmp[i] = (double)(i + 1);
    }
    cudaMemcpy(arrayA, tmp, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(arrayB, tmp, arrSize, cudaMemcpyHostToDevice);

    for (size_t i = 0; i < N; i++) {
        tmp[i] = (double)(N - i);
    }
    cudaMemcpy(arrayC, tmp, arrSize, cudaMemcpyHostToDevice);

    for (size_t i = 0; i < N; i++) {
        tmp[i] = 0;
    }
    cudaMemcpy(arrayEmpty, tmp, arrSize, cudaMemcpyHostToDevice);

    for (size_t i = 0; i < N; i++) {
        tmp[i] = 1;
    }
    cudaMemcpy(arrayOne, tmp, arrSize, cudaMemcpyHostToDevice);

    free(tmp);
}

void cleanTestStructures() {
    cudaFree(arrayA);
    cudaFree(arrayB);
    cudaFree(arrayC);
    cudaFree(arrayEmpty);
    cudaFree(arrayOne);
    cudaFree(device_result);
    free(result);
}

void testDotProduct() {
    double expected;

    // Dot product of any array with the zero vector is zeroes.
    expected = 0;
    vector_dot_product<<<1, N, arrSize>>>(N, arrayA, arrayEmpty, device_result);
    cudaMemcpy(result, device_result, sizeof(double), cudaMemcpyDeviceToHost);
    printf("arrayA dotted with the zero vector should be: %f ; got: %f\n", expected,
           *result);
    if (fabs(expected - *result) >= FEPSILON) {
        printf("Failed!\n");
        exit(1);
    }

    // Dot product of an array with all 1's should be the sum of elements in
    // the original array.
    // Sum of 1..N should be (N*(N+1)/2)
    expected = N * (N + 1) / 2;
    vector_dot_product<<<1, N, arrSize>>>(N, arrayA, arrayOne, device_result);
    cudaMemcpy(result, device_result, sizeof(double), cudaMemcpyDeviceToHost);
    printf("arrayA dotted with arrayOne should be: %f ; got: %f\n", expected, *result);
    if (fabs(expected - *result) >= FEPSILON) {
        printf("Failed!\n");
        exit(1);
    }
    // Dot product of an array with itself can be calculated manually.
    expected = 0;
    for (size_t i = 0; i < N; i++)
        expected += (i + 1) * (i + 1);
    vector_dot_product<<<1, N, arrSize>>>(N, arrayA, arrayA, device_result);
    cudaMemcpy(result, device_result, sizeof(double), cudaMemcpyDeviceToHost);
    printf("arrayA dotted with arrayA should be: %f ; got: %f\n", expected, *result);
    if (fabs(expected - *result) >= FEPSILON) {
        printf("Failed!\n");
        exit(1);
    }
}

void testSubtraction() {

    double *arr_result = (double *)malloc(arrSize);
    double  expected;

    // An array minus the zero vector should be the original array
    vector_subtraction<<<1, N>>>(N, arrayEmpty, arrayA);
    cudaMemcpy(arr_result, arrayEmpty, arrSize, cudaMemcpyDeviceToHost);
    printf("the zero vector - arrayA\n");
    for (size_t i = 0; i < N; i++) {
        expected = -(((double)i) + 1.);
        if (fabs(expected - arr_result[i]) >= FEPSILON) {
            printf("Failed vector subtraction. expected: %f ; got: %f\n", expected,
                   arr_result[i]);
            exit(1);
        }
    }

    // An array minus the one vector
    vector_subtraction<<<1, N>>>(N, arrayOne, arrayA);
    cudaMemcpy(arr_result, arrayOne, arrSize, cudaMemcpyDeviceToHost);
    printf("arrayA - the one vector\n");
    for (size_t i = 0; i < N; i++) {
        expected = -((double)i);
        if (fabs(expected - arr_result[i]) >= FEPSILON) {
            printf("Failed vector subtraction. expected: %f ; got: %f\n", expected,
                   arr_result[i]);
            exit(1);
        }
    }

    // An array minus itself
    vector_subtraction<<<1, N>>>(N, arrayB, arrayA);
    cudaMemcpy(arr_result, arrayB, arrSize, cudaMemcpyDeviceToHost);
    printf("arrayA - arrayB\n");
    expected = 0;
    for (size_t i = 0; i < N; i++) {
        if (fabs(expected - arr_result[i]) >= FEPSILON) {
            printf("Failed vector subtraction. expected: %f ; got: %f\n", expected,
                   arr_result[i]);
            exit(1);
        }
    }

    // Throughout this process, arrayA should not have changed.
    cudaMemcpy(arr_result, arrayA, arrSize, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < N; i++) {
        expected = i + 1;
        if (fabs(expected - arr_result[i]) >= FEPSILON) {
            printf(
                "Failed. Vector subtraction should not affect the subtrahend. expected: "
                "%f ; got: %f\n",
                expected, arr_result[i]);
            exit(1);
        }
    }
}

void testManySubtractionFunction() {
    // Test many_vector_subtraction
    // Create A = [ \vec 0 & \vec 0 ... & \vec 0 ]
    // Create base = \vec 1
    // Create coefficients = [ 1 & 2 ... & n ]
    // Assert that A = [ \vec -1 & \vec -2 ... & \vec -n ]
    // Vectors have size nx1, matrices have size nxm
    const size_t M = N * 2;

    double **A, **Arepr, *tmp, *tmprepr;
    cudaMalloc(&A, sizeof(double *) * M);
    Arepr = (double **)malloc(sizeof(double *) * M);

    // Set up that zero vector
    tmprepr = (double *)malloc(sizeof(double) * N);
    for (size_t i = 0; i < N; i++)
        tmprepr[i] = 0;

    for (size_t i = 0; i < M; i++) {
        cudaMalloc(&tmp, sizeof(double) * N);
        // Copy the zero vector in everywhere
        cudaMemcpy(tmp, tmprepr, sizeof(double) * N, cudaMemcpyHostToDevice);
        Arepr[i] = tmp;
    }
    cudaMemcpy(A, Arepr, sizeof(double) * M, cudaMemcpyHostToDevice);

    // Create the coefficients
    double *coefficients, *coefficientsrepr;
    cudaMalloc(&coefficients, sizeof(double) * M);
    coefficientsrepr = (double *)malloc(sizeof(double) * M);
    for (size_t i = 0; i < M; i++)
        coefficientsrepr[i] = (i + 1);
    cudaMemcpy(coefficients, coefficientsrepr, sizeof(double) * M,
               cudaMemcpyHostToDevice);

    // Create the 'base' vector of 1s
    double *base;
    cudaMalloc(&base, sizeof(double) * N);
    for (size_t i = 0; i < N; i++)
        tmprepr[i] = 1;
    cudaMemcpy(base, tmprepr, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Run the code:
    printf("Testing 'many_vector_subtractions'\n");
    many_vector_subtractions<<<1, N>>>(base, A, coefficients, M, N);

    // Look at the vectors in A and ensure they are as expected
    cudaMemcpy(Arepr, A, sizeof(double) * M, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < M; i++) {
        cudaMemcpy(tmprepr, Arepr[i], sizeof(double) * N, cudaMemcpyDeviceToHost);
        double expected = -(((double)i) + 1.);
        for (size_t j = 0; j < N; j++) {
            if (fabs(tmprepr[j] - expected) >= FEPSILON) {
                printf(
                    "Failed. Subtract Many Vectors index %lu should be: %f ; got: %f\n",
                    i, expected, tmprepr[j]);
                exit(1);
            }
        }
    }
}

void testManyDotProductFunction() {
    // Test many_vector_dot_product
    // Create A = [ arrayA & arrayC & arrayEmpty & arrayOne ]
    // Create base = \vec 1
    // Assert that A = [ (arrayA sum) & (arrayA sum) & 0 & n ]
    // Vectors have size nx1, matrices have size nx4, output should be 1x4
    const size_t M = 4;

    double **A, **Arepr;
    cudaMalloc(&A, sizeof(double *) * M);
    Arepr = (double **)malloc(sizeof(double *) * M);
    Arepr[0] = arrayA;
    Arepr[1] = arrayC;
    Arepr[2] = arrayEmpty;
    Arepr[3] = arrayOne;
    cudaMemcpy(A, Arepr, sizeof(double) * M, cudaMemcpyHostToDevice);

    double *base;
    cudaMalloc(&base, sizeof(double *) * N);
    cudaMemcpy(base, arrayOne, sizeof(double) * N, cudaMemcpyDeviceToDevice);

    double *result, *resultrepr;
    cudaMalloc(&result, sizeof(double) * M);
    resultrepr = (double *)malloc(sizeof(double) * M);

    // Run the code:
    printf("Testing 'many_vector_dot_product'\n");
    many_vector_dot_product<<<1, N, N * M * sizeof(double)>>>(base, A, M, N, result);

    // Validate all 4 results
    cudaMemcpy(resultrepr, result, sizeof(double) * M, cudaMemcpyDeviceToHost);

    double expected, found;
    expected = N * (N + 1) / 2;
    found = resultrepr[0];
    if (fabs(found - expected) >= FEPSILON) {
        printf("Failed. many_vector_dot_product index 0 should be: %f ; got: %f\n",
               expected, found);
        exit(1);
    }

    // same expected
    found = resultrepr[1];
    if (fabs(found - expected) >= FEPSILON) {
        printf("Failed. many_vector_dot_product index 1 should be: %f ; got: %f\n",
               expected, found);
        exit(1);
    }

    expected = 0;
    found = resultrepr[2];
    if (fabs(found - expected) >= FEPSILON) {
        printf("Failed. many_vector_dot_product index 2 should be: %f ; got: %f\n",
               expected, found);
        exit(1);
    }

    expected = N;
    found = resultrepr[3];
    if (fabs(found - expected) >= FEPSILON) {
        printf("Failed. many_vector_dot_product index 3 should be: %f ; got: %f\n",
               expected, found);
        exit(1);
    }
}

int main() {
    // --------------------
    createTestStructures();
    testDotProduct();
    cleanTestStructures();
    // --------------------
    createTestStructures();
    testSubtraction();
    cleanTestStructures();
    // --------------------
    testManySubtractionFunction();
    // --------------------
    createTestStructures();
    testManyDotProductFunction();
    cleanTestStructures();
    // --------------------
    printf("\n\nAll tests passed.\n");
    return 0;
}
