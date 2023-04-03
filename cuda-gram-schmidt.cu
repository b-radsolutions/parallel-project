//
// CUDA CODE FOR MODIFIED GRAM SCHMIDT
//


/*
    * vector_subtraction
    * @PARM n size of smaller array
    * @PARM *x vector of minuend
    * @PARM *y vector of subtrahend
    * @REQUIRES *x and *y be equal in size
    * @REQUIRES *x and *y be a pointer in device memory
    * @MODIFIES *y
    * @EFFECTS *y[i] is the difference x[i] - y[i]
    *
*/
__global__ void vector_subtraction(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] - y[i];
}


/*
    * vector_projection
    * @PARM n size of vector
    * @PARM *x vector
    * @PARM *y vector
    * @REQUIRES *x and *y be equal in size
    * @REQUIRES *x and *y be a pointer in device memory
    * @RETURNS int that dot product
    *
*/
__global__ int vector_dot_product(int n, float *x, float *y) {
    __shared__ int temp[n];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[index] = x[index] * y[index];

    __syncthreads();

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += temp[i];
        }
        return sum;
    }
}


