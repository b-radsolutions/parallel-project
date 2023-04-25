//
// CUDA CODE FOR MODIFIED GRAM SCHMIDT
//

#include "matrix-operations.hpp"

#include "math.h"
#include "mpi-helper.hpp"
#include <cstdlib>
#include <iostream>
#include <stdio.h>

#define calc1dIndex blockIdx.x *blockDim.x + threadIdx.x

/*
 * vector_subtraction
 * @PARM n size of both arrays
 * @PARM *x vector of minuend
 * @PARM *y vector of subtrahend
 * @REQUIRES *x and *y be equal in size
 * @REQUIRES *x and *y be a pointer in device memory
 * @MODIFIES *x
 * @EFFECTS *x[i] is the difference x[i] - y[i]
 */
__global__ void vector_subtraction(int n, double *x, double *y) {
    int index = calc1dIndex;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = x[i] - y[i];
}

/*
 * vector_normalize
 * @PARM n size of array
 * @PARM *x vector array
 * @PARM y number to normalize by
 * @REQUIRES *x and *y to be a pointer in device memory
 * @MODIFIES *x
 * @EFFECTS *x[i] is now x[i] / y
 */
__global__ void vector_normalize(int n, double *x, double *y) {
    double denom = sqrt(*y);
    int    index = calc1dIndex;
    int    stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = x[i] / denom;
}

/*
 * vector_normalize
 * @PARM n size of array
 * @PARM *x vector array
 * @PARM y number to mult by
 * @REQUIRES *x and *y to be a pointer in device memory
 * @MODIFIES *x
 * @EFFECTS *x[i] is now x[i] * y
 */
__global__ void vector_mult(int n, double *x, double *y) {
    int index = calc1dIndex;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = x[i] * *y;
}

/*
 * vector_projection
 * @PARM n size of both vectors
 * @PARM *x vector
 * @PARM *y vector
 * @PARM *result a pointer to a single double value in which the result will be stored.
 * @REQUIRES *x and *y be equal in size
 * @REQUIRES *x and *y be a pointer in device memory
 */
__global__ void vector_dot_product(int n, double *x, double *y, double *result) {
    extern __shared__ double temp[];

    int index = calc1dIndex;
    temp[index] = x[index] * y[index];

    __syncthreads();

    if (index == 0) {
        *result = 0;
        for (int i = 0; i < n; i++) {
            *result += temp[i];
        }
    }
}

/**
 * Subtract the base vector from every other vector using the magnitude in scalars for the
 * base vector.
 * V = V - CB where v = vectors, b = base, and C is scalars.
 * len(result) == len(vectors) == len(scalars) == num_vectors ;
 * len(vectors[:]) == len(base) == n
 */
__global__ void many_vector_subtractions(double *base, double **vectors, double *scalars,
                                         size_t num_vectors, size_t n) {

    int    index = calc1dIndex;
    int    stride = blockDim.x * gridDim.x;
    size_t total = num_vectors * n;

    for (int i = index; i < total; i += stride) {
        int block = i / n;
        int vi = i % n;
        vectors[block][vi] -= base[vi] * scalars[block];
    }
    __syncthreads();
}

/**
 * Dot the base with every other vector, returning the individual dot products in result.
 * len(result) == len(vectors) == num_vectors ;
 * len(vectors[:]) == len(base) == n
 */
__global__ void many_vector_dot_product(double *base, double **vectors,
                                        size_t num_vectors, size_t n, double *result) {
    extern __shared__ double temp[];

    int    index = calc1dIndex;
    int    stride = blockDim.x * gridDim.x;
    size_t total = num_vectors * n;

    for (int i = index; i < total; i += stride) {
        int block = i / n;
        int vi = i % n;
        temp[i] = vectors[block][vi] * base[vi];
    }

    __syncthreads();

    // Perform the reduction
    if (index < num_vectors) {
        int    start = index * n;
        int    end = start + n;
        double res = 0;
        for (int i = start; i < end; i++) {
            res += temp[i];
        }
        result[index] = res;
    }
    __syncthreads();
}

// ----------------------------------------
// Cuda Entry Points
// ----------------------------------------

double *magnitude;
double *host_magnitude;

void cudaSetup(size_t myrank) {
    int cE; // Cuda Error
    int cudaDeviceCount;
    if ((cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess) {
        std::cout << " Unable to determine cuda device count, error is " << cE
                  << ", count is " << cudaDeviceCount << "\n";
        exit(-1);
    }
    if ((cE = cudaSetDevice(myrank % cudaDeviceCount)) != cudaSuccess) {
        std::cout << " Unable to have rank " << myrank << " set to cuda device "
                  << (myrank % cudaDeviceCount) << ", error is " << cE << " \n";
        exit(-1);
    }

    // Set up memory to hold the result of dot product
    cudaMalloc(&magnitude, sizeof(double));
    host_magnitude = (double *)malloc(sizeof(double));
}

void cudaCleanup() { cudaFree(magnitude); }

void cleanupMatrix(double **A, size_t number_vectors) {
    for (size_t i = 0; i < number_vectors; i++)
        cudaFree(A[i]);
    free(A);
}

void cleanupVector(double *vec) { cudaFree(vec); }

// Create 'n' random columns of 'n' entries
double **createTestMatrix(size_t number_vectors, size_t vector_size) {
    double **ret, *tmp, *local;

    // Create local to hold the randomly-generated column
    local = (double *)malloc(sizeof(double) * vector_size);

    // ret will be created on the CPU so it can reference the devices pointers
    ret = (double **)malloc(sizeof(double *) * number_vectors);

    for (size_t i = 0; i < number_vectors; i++) {
        // Randomly populate local copy
        for (size_t j = 0; j < vector_size; j++) {
            local[j] = ((double)rand() / (double)RAND_MAX);
        }
        // Transfer local copy onto the device
        cudaMalloc(&tmp, sizeof(double) * vector_size);
        cudaMemcpy(tmp, local, sizeof(double) * vector_size, cudaMemcpyHostToDevice);
        // Set the row
        ret[i] = tmp;
    }

    free(local);

    return ret;
}

double **allocateMatrix(size_t number_vectors, size_t vector_size) {
    double **ret, *tmp;
    // ret will be created on the CPU so it can reference the device pointers
    ret = (double **)malloc(sizeof(double *) * number_vectors);
    for (size_t i = 0; i < number_vectors; i++) {
        // Transfer local copy onto the device
        cudaMalloc(&tmp, sizeof(double) * vector_size);
        // Set the row
        ret[i] = tmp;
    }
    return ret;
}

// double **allocateMNMatrix(size_t n, size_t m) {
//     double **ret, *tmp;
//     // ret will be created on the CPU so it can reference the device pointers
//     ret = (double **)malloc(sizeof(double *) * m);
//     for (size_t i = 0; i < m; i++) {
//         // Transfer local copy onto the device
//         cudaMalloc(&tmp, sizeof(double) * n);
//         // Set the row
//         ret[i] = tmp;
//     }
//     return ret;
// }

double *allocateVector(size_t vector_size) {
    double *ret;
    cudaMalloc(&ret, sizeof(double) * vector_size);
    return ret;
}

double **matrixDeviceToHost(double **A, size_t number_vectors, size_t vector_size) {
    double *tmp, **ret = (double **)malloc(sizeof(double *) * number_vectors);
    for (size_t i = 0; i < number_vectors; i++) {
        tmp = (double *)malloc(sizeof(double) * vector_size);
        cudaMemcpy(tmp, A[i], vector_size * sizeof(double), cudaMemcpyDeviceToHost);
        ret[i] = tmp;
    }
    return ret;
}

double **matrixHostToDevice(double **A, size_t number_vectors, size_t vector_size) {
    double **ret, *tmp;
    ret = (double **)malloc(sizeof(double *) * number_vectors);
    for (size_t i = 0; i < number_vectors; i++) {
        // Transfer local copy onto the device
        cudaMalloc(&tmp, sizeof(double) * vector_size);
        // Copy the memory over
        cudaMemcpy(tmp, A[i], vector_size * sizeof(double), cudaMemcpyHostToDevice);
        // Set the row
        ret[i] = tmp;
    }
    return ret;
}

void matrixCopy(double **A, double **B, size_t number_vectors, size_t vector_size) {
    for (size_t i = 0; i < number_vectors; i++) {
        cudaMemcpy(B[i], A[i], vector_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }
}

// Returns the reduced result of the dot product operation
double distributedDotProduct(double *a, double *b, size_t partial_vector_size) {
    double res;
    vector_dot_product<<<1, partial_vector_size, sizeof(double) * partial_vector_size>>>(
        partial_vector_size, a, b, magnitude);
    cudaMemcpy(host_magnitude, magnitude, sizeof(double), cudaMemcpyDeviceToHost);
    my_MPIReduce(host_magnitude, 1, &res);
    return res;
}

void distributedNormalize(double *src, double *dst, size_t partial_vector_size) {
    double dot = distributedDotProduct(src, src, partial_vector_size);
    cudaMemcpy(magnitude, &dot, sizeof(double), cudaMemcpyHostToDevice);
    // Need to copy the src into the dst before we divide if different.
    if (src != dst) {
        cudaMemcpy(dst, src, sizeof(double) * partial_vector_size,
                   cudaMemcpyDeviceToDevice);
    }
    // Divide happens here.
    vector_normalize<<<1, partial_vector_size>>>(partial_vector_size, dst, magnitude);
}

void normalize(double *src, double *dst, size_t vector_size) {
    // Find the value to divide by
    vector_dot_product<<<1, vector_size, sizeof(double) * vector_size>>>(vector_size, src,
                                                                         src, magnitude);
    if (src != dst) {
        // Need to copy the src into the dst before we divide
        cudaMemcpy(dst, src, sizeof(double) * vector_size, cudaMemcpyDeviceToDevice);
    }
    // Divide happens here
    vector_normalize<<<1, vector_size>>>(vector_size, dst, magnitude);
}

void distributedProjection(double *vector, double *base, double *result,
                           size_t partial_vector_size) {
    // We assume the base to have magnitude 1, saving us from this division.
    // But we do need to find the numerator.
    double dot = distributedDotProduct(vector, base, partial_vector_size);
    if (base != result) {
        // Need to copy the base to the result before we multiply, as it happens in-place.
        cudaMemcpy(result, base, sizeof(double) * partial_vector_size,
                   cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(magnitude, &dot, sizeof(double), cudaMemcpyHostToDevice);
    // Now, we can multiply the base by this magnitude
    vector_mult<<<1, partial_vector_size>>>(partial_vector_size, result, magnitude);
}

// Requires the base to have magnitude 1 (to avoid an extra dot product)
void projection(double *vector, double *base, double *result, size_t vector_size) {
    // Find the numerator for the projection quotient
    vector_dot_product<<<1, vector_size, sizeof(double) * vector_size>>>(
        vector_size, vector, base, magnitude);
    // We assume the base to have magnitude 1, saving us from this division
    if (base != result) {
        // Need to copy the base to the result before we multiply, as it happens in-place.
        cudaMemcpy(result, base, sizeof(double) * vector_size, cudaMemcpyDeviceToDevice);
    }
    // Now, we can multiply the base by this magnitude
    vector_mult<<<1, vector_size>>>(vector_size, result, magnitude);
}

void subtract(double *a, double *b, double *dst, size_t vector_size) {
    if (a != dst) {
        cudaMemcpy(dst, a, sizeof(double) * vector_size, cudaMemcpyDeviceToDevice);
    }
    vector_subtraction<<<1, vector_size>>>(vector_size, dst, b);
}

double dot(double *a, double *b, size_t vector_size) {
    // Take the dot product
    vector_dot_product<<<1, vector_size, sizeof(double) * vector_size>>>(vector_size, a,
                                                                         b, magnitude);
    // Need to take the result out of device memory
    cudaMemcpy(host_magnitude, magnitude, sizeof(double), cudaMemcpyDeviceToHost);
    return *host_magnitude;
}

// Removes the projection of the completed index from every vector afterwards. A
// has `m` columns and `n` rows.
void performModifiedGramSchmidtReduction(double **A, size_t number_vectors,
                                         size_t partial_vector_size,
                                         size_t completed_index) {
    double  *base = A[completed_index];
    size_t   remainder_count = number_vectors - (completed_index + 1);
    double **remainder = (A + completed_index + 1);
    size_t   coefficient_size = sizeof(double) * remainder_count;
    double  *dots, *host_dots, *coeffs;

    cudaMalloc(&dots, coefficient_size);

    host_dots = (double *)malloc(coefficient_size);
    coeffs = (double *)malloc(coefficient_size);

    many_vector_dot_product<<<1, partial_vector_size,
                              sizeof(double) * remainder_count * partial_vector_size>>>(
        base, remainder, remainder_count, partial_vector_size, dots);

    // Communicate with the other MPI ranks to discover the complete dot product
    cudaMemcpy(host_dots, dots, coefficient_size, cudaMemcpyDeviceToHost);
    my_MPIReduce(host_dots, remainder_count, coeffs);
    cudaMemcpy(dots, coeffs, coefficient_size, cudaMemcpyHostToDevice);

    // Use that dot product to do vector subtractions
    many_vector_subtractions<<<1, partial_vector_size>>>(
        base, remainder, dots, remainder_count, partial_vector_size);

    free(host_dots);
    free(coeffs);
    cudaFree(dots);
}
