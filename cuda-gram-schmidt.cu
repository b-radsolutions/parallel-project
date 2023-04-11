//
// CUDA CODE FOR MODIFIED GRAM SCHMIDT
//

#include <cstdlib>

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
__global__ void vector_subtraction(int n, float *x, float *y) {
    int index = calc1dIndex;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = x[i] - y[i];
}

/*
 * vector_projection
 * @PARM n size of both vectors
 * @PARM *x vector
 * @PARM *y vector
 * @PARM *result a pointer to a single float value in which the result will be stored.
 * @REQUIRES *x and *y be equal in size
 * @REQUIRES *x and *y be a pointer in device memory
 */
__global__ void vector_dot_product(int n, float *x, float *y, float *result) {
    extern __shared__ float temp[];

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

// ----------------------------------------
// Cuda Entry Points
// ----------------------------------------

float *magnitude;

void cudaSetup() {
    // Set up memory to hold the result of dot product
    cudaMalloc(&magnitude, sizeof(double));
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
        cudaMemcpy(tmp, local, sizeof(double) * n, cudaMemcpyHostToDevice);
        // Set the row
        ret[i] = tmp;
    }
    return ret;
}

void normalize(double *src, double *dst, size_t n) {
    // Find the value to divide by
    vector_dot_product<<<1, n, sizeof(double) * n>>>(n, src, src, magnitude);
    if (src != dst) {
        // Need to copy the src into the dst before we divide
        cudaMemcpy(dst, src, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    }
    // Divide happens here
    // todo:: add the divide
}

// Requires the base to have magnitude 1 (to avoid an extra dot product)
void projection(double *vector, double *base, double *result, size_t n) {
    // Find the numerator for the projection quotient
    vector_dot_product<<<1, n, sizeof(double) * n>>>(n, src, src, magnitude);
    // We assume the base to have magnitude 1, saving us from this division
    if (base != result) {
        // Need to copy the base to the result before we multiply, as it happens in-place.
        cudaMemcpy(result, base, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    }
    // Now, we can multiply the base by this magnitude
    // todo:: add the multiplication
}

void subtract(double *a, double *b, double *dst, size_t n) {
    if (a != dst) {
        cudaMemcpy(dst, a, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    }
    vector_subtraction<<<1, n, sizeof(double) * n>>>(n, dst, b);
}
