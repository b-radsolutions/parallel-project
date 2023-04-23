//
// CUDA CODE FOR MODIFIED GRAM SCHMIDT
//

#include "math.h"
#include <cstdlib>

#include "mpi-helper.hpp"

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
}

// ----------------------------------------
// Cuda Entry Points
// ----------------------------------------

double *magnitude;
double *host_magnitude;

void cudaSetup() {
    // Set up memory to hold the result of dot product
    cudaMalloc(&magnitude, sizeof(double));
    host_magnitude = (double *)malloc(sizeof(double));
}

void cudaCleanup() { cudaFree(magnitude); }

void cleanupMatrix(double **A, size_t m) {
    for (size_t i = 0; i < m; i++)
        cudaFree(A[i]);
    free(A);
}

// Create 'n' random columns of 'n' entries
double **createTestMatrix(size_t n) {
    double **ret, *tmp, *local;

    // Create local to hold the randomly-generated column
    local = (double *)malloc(sizeof(double) * n);

    // ret will be created on the CPU so it can reference the devices pointers
    ret = (double **)malloc(sizeof(double *) * n);

    for (size_t i = 0; i < n; i++) {
        // Randomly populate local copy
        for (size_t j = 0; j < n; j++) {
            local[j] = ((double)rand() / (double)RAND_MAX);
        }
        // Transfer local copy onto the device
        cudaMalloc(&tmp, sizeof(double) * n);
        cudaMemcpy(tmp, local, sizeof(double) * n, cudaMemcpyHostToDevice);
        // Set the row
        ret[i] = tmp;
    }

    free(local);

    return ret;
}

double **allocateMatrix(size_t n) {
    double **ret, *tmp;
    // ret will be created on the CPU so it can reference the device pointers
    ret = (double **)malloc(sizeof(double *) * n);
    for (size_t i = 0; i < n; i++) {
        // Transfer local copy onto the device
        cudaMalloc(&tmp, sizeof(double) * n);
        // Set the row
        ret[i] = tmp;
    }
    return ret;
}

void matrixCopy(double **A, double **B, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        cudaMemcpy(B[i], A[i], n * sizeof(double), cudaMemcpyDeviceToDevice);
    }
}

void normalize(double *src, double *dst, size_t n) {
    // Find the value to divide by
    vector_dot_product<<<1, n, sizeof(double) * n>>>(n, src, src, magnitude);
    if (src != dst) {
        // Need to copy the src into the dst before we divide
        cudaMemcpy(dst, src, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    }
    // Need to take the result out of device memory to reduce it
    cudaMemcpy(host_magnitude, magnitude, sizeof(double), cudaMemcpyDeviceToHost);
    double res;
    my_MPIReduce(host_magnitude, 1, &res);
    cudaMemcpy(magnitude, &res, sizeof(double), cudaMemcpyHostToDevice);
    // Divide happens here
    vector_normalize<<<1, n>>>(n, dst, magnitude);
}

// Requires the base to have magnitude 1 (to avoid an extra dot product)
void projection(double *vector, double *base, double *result, size_t n) {
    // Find the numerator for the projection quotient
    vector_dot_product<<<1, n, sizeof(double) * n>>>(n, vector, base, magnitude);
    // We assume the base to have magnitude 1, saving us from this division
    if (base != result) {
        // Need to copy the base to the result before we multiply, as it happens in-place.
        cudaMemcpy(result, base, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    }
    // Need to take the result out of device memory to reduce it
    cudaMemcpy(host_magnitude, magnitude, sizeof(double), cudaMemcpyDeviceToHost);
    double res;
    my_MPIReduce(host_magnitude, 1, &res);
    cudaMemcpy(magnitude, &res, sizeof(double), cudaMemcpyHostToDevice);
    // Now, we can multiply the base by this magnitude
    vector_mult<<<1, n>>>(n, result, magnitude);
}

void subtract(double *a, double *b, double *dst, size_t n) {
    if (a != dst) {
        cudaMemcpy(dst, a, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    }
    vector_subtraction<<<1, n>>>(n, dst, b);
}

double dot(double *a, double *b, size_t n) {
    // Take the dot product
    vector_dot_product<<<1, n, sizeof(double) * n>>>(n, a, b, magnitude);
    // Need to take the result out of device memory
    cudaMemcpy(host_magnitude, magnitude, sizeof(double), cudaMemcpyDeviceToHost);
    return *host_magnitude;
}

// Removes the projection of the completed index from every vector afterwards. A
// has `m` columns and `n` rows.
void performModifiedGramSchmidtReduction(double **A, size_t m, size_t n,
                                         size_t completed_index) {
    double  *base = A[completed_index];
    size_t   remainder_count = m - (completed_index + 1);
    double **remainder = (A + completed_index + 1);
    size_t   coefficient_size = sizeof(double) * remainder_count;
    double  *dots, *host_dots, *coeffs;
    cudaMalloc(&dots, coefficient_size);
    host_dots = (double *)malloc(coefficient_size);
    coeffs = (double *)malloc(coefficient_size);
    // many_vector_projection_subtractions<<<1, n>>>(base, remainder, remainder_count,
    // n); Do all of the dot products
    many_vector_dot_product<<<1, n, coefficient_size>>>(base, remainder, remainder_count,
                                                        n, dots);
    // Communicate with the other MPI ranks to discover the complete dot product
    cudaMemcpy(host_dots, dots, coefficient_size, cudaMemcpyDeviceToHost);
    my_MPIReduce(host_dots, remainder_count, coeffs);
    cudaMemcpy(dots, coeffs, coefficient_size, cudaMemcpyHostToDevice);
    // Use that dot product to do vector subtractions
    many_vector_subtractions<<<1, n>>>(base, remainder, dots, remainder_count, n);
    cudaFree(dots);
}
