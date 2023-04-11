
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10
#define GRID_SIZE 16
#define FEPSILON 0.0001

__global__ void vector_subtraction(int n, float *x, float *y);
__global__ void vector_dot_product(int n, float *x, float *y, float *result);

const size_t arrSize = N * sizeof(double);

float *arrayA;     // 1, 2, 3, ... n-1, n
float *arrayB;     // 1, 2, 3, ... n-1, n
float *arrayC;     // n, n-1, ... 3, 2, 1
float *arrayEmpty; // 0, 0, 0, ... 0
float *arrayOne;   // 1, 1, 1, ... 1
float *device_result;
float *result;

void createTestStructures() {
    cudaMalloc(&arrayA, arrSize);
    cudaMalloc(&arrayB, arrSize);
    cudaMalloc(&arrayC, arrSize);
    cudaMalloc(&arrayEmpty, arrSize);
    cudaMalloc(&arrayOne, arrSize);
    cudaMalloc(&device_result, sizeof(double));
    result = (float *)malloc(sizeof(double));

    float *tmp = (float *)malloc(arrSize);
    for (size_t i = 0; i < N; i++) {
        tmp[i] = (float)(i + 1);
    }
    cudaMemcpy(arrayA, tmp, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(arrayB, tmp, arrSize, cudaMemcpyHostToDevice);

    for (size_t i = 0; i < N; i++) {
        tmp[i] = (float)(N - i);
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
    float expected;

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

    float *arr_result = (float *)malloc(arrSize);
    float  expected;

    // An array minus the zero vector should be the original array
    vector_subtraction<<<1, N>>>(N, arrayA, arrayEmpty);
    cudaMemcpy(arr_result, arrayEmpty, arrSize, cudaMemcpyDeviceToHost);
    printf("arrayA - the zero vector\n");
    for (size_t i = 0; i < N; i++) {
        expected = i + 1;
        if (fabs(expected - arr_result[i]) >= FEPSILON) {
            printf("Failed vector subtraction. expected: %f ; got: %f\n", expected,
                   arr_result[i]);
            exit(1);
        }
    }

    // An array minus the one vector
    vector_subtraction<<<1, N>>>(N, arrayA, arrayOne);
    cudaMemcpy(arr_result, arrayOne, arrSize, cudaMemcpyDeviceToHost);
    printf("arrayA - the one vector\n");
    for (size_t i = 0; i < N; i++) {
        expected = i;
        if (fabs(expected - arr_result[i]) >= FEPSILON) {
            printf("Failed vector subtraction. expected: %f ; got: %f\n", expected,
                   arr_result[i]);
            exit(1);
        }
    }

    // An array minus itself
    vector_subtraction<<<1, N>>>(N, arrayA, arrayB);
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
            printf("Failed. Vector subtraction should not affect the minuend. expected: "
                   "%f ; got: %f\n",
                   expected, arr_result[i]);
            exit(1);
        }
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
    printf("\n\nAll tests passed.\n");
    return 0;
}
