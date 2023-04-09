//
// CUDA CODE FOR MODIFIED GRAM SCHMIDT
//


/*
    * vector_subtraction
    * @PARM n size of both arrays
    * @PARM *x vector of minuend
    * @PARM *y vector of subtrahend
    * @REQUIRES *x and *y be equal in size
    * @REQUIRES *x and *y be a pointer in device memory
    * @MODIFIES *y
    * @EFFECTS *y[i] is the difference x[i] - y[i]
*/
__global__ void vector_subtraction(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] - y[i];
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
    __shared__ float temp[n];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[index] = x[index] * y[index];

    __syncthreads();

    if (index == 0) {
        *result = 0;
        for (int i = 0; i < n; i++)
        {
            *result += temp[i];
        }
    }
}


