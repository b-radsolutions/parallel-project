#include <cstdlib.h>
#include <stddef.h>
#include <string.h>
#include <cmath>

void normalize(double *src, double *dst, size_t n);
void projection(double *vector, double *base, double *result, size_t n);
void subtract(double *a, double *b, double *dst, size_t n);
void dot(double *a, double *b, double *dst, size_t n);
void transpose(double **Α, double **ΑT, size_t m, size_t n);
void matrix_mult(double **Α, size_t m, size_t n_m, double **B, size_t n1, double **output);
void matrix_sub(double **Α, size_t m, size_t n_m, double **B, size_t n1, double **output);
double** eye(n);

//Test for orthogonality Frobenius norm
// concats matrix rows then finds the
// 2-norm of the vector in R^(m*n)
double frobeniusNorm(size_t m, size_t n, double **Q){
    //matrix multiplication to get Q^T * Q - I = E
    //where E is the error matrix
    double **output = (double *)malloc(sizeof(double) * n);
    matrix_mult(Q, m, n, Q, n, output);
    double **I = eye(m);
    matrix_sub(output, m, n, I, n, output)
    double total = 0.0;
    for(size_t i = 0; i < m; i++) { 
        for(size_t j = 0; j < m; j++) { 
            total += output[i][j] * output[i][j];
        }   
    }
    return sqrt(total);
}

//Testing suite to gauge how orthogonal a matrix A is.
//takes the infinity norm of the errors of each column in Q
//and then takes the 1-norm of each vector inf-norm
double inf_norm_test(size_t m, size_t n, double **Q){
    double error = 0.0;
    double *tmp;
    for(size_t i = 0; i < m; i++) { 
        double max_dot = 0.0;
        for(size_t j = 0; j < m; j++) { 
            if(i != j) {
                dot(Q[j], Q[i], tmp, n);
                total += *tmp;
                if(*tmp > max_dot){
                    max_dot = *tmp;
                }
            }
        }
        error += max_dot;  
    }
    return error;
}

//Testing suite to gauge how orthogonal a matrix A is.
//takes the 2-norm of of the errors of each
double one_norm_test(size_t m, size_t n, double **Q){
    double *dot_totals = (double *)malloc(sizeof(double) * n);
    double *tmp;
    for(size_t i = 0; i < m; i++) { 
        double total = 0.0;
        for(size_t j = 0; j < m; j++) { 
            if(i != j) {
                dot(Q[j], Q[i], tmp, n);
                total += *tmp * *tmp;
            }
        }
        
        dot_totals[i] = total;
        dot_maxes[i] = max_dot;       
    }
}

//Testing suite to gauge how orthogonal a matrix A is.
// 
double* orthogonality_test1(size_t m, size_t n, double **Q,  double *dot_totals, double *dot_maxes){
    double *tmp;
    for(size_t i = 0; i < m; i++) { 
        double total = 0.0
        double max_dot = 0.0;
        for(size_t j = 0; j < m; j++) { 
            if(i != j) {
                dot(Q[j], Q[i], tmp, n);
                total += *tmp;
                if(tmp > max_dot){
                    max_dot = *tmp;
                }
            }
        }
        dot_totals[i] = total;
        dot_maxes[i] = max_dot;       
    }
    free(tmp);
}




// For an m x n matrix A (A[m][n] is the bottom right entry, A has m columns with n rows
// each), orthonormalize the matrix A and put the result in the pre-allocated Q.
// **Note** : This matrix indexing is intentionally different than usual matrix
// indexing since we are doing operations mainly by column.
void classical_gram_schmidt(double **A, size_t m, size_t n, double **Q) {
    // Our first vector is already done
    normalize(Q[0], Q[0], n);

    // Copy over the rest of A into the output Q
    for (size_t i = 1; i < m; i++) {
        memcpy(Q[i], A[i], sizeof(double) * n);
    }

    double *tmp = (double *)malloc(sizeof(double) * n);
    for (size_t i = 0; i < m - 1; i++) {
        // Subtract the projection of the previously-completed vector from the remaining
        for (size_t j = i + 1; j < m; j++) {
            projection(Q[j], Q[i], tmp, n);
            subtract(Q[j], tmp, Q[j], n);
        }
        // Normalize the vector we just completed
        normalize(Q[i + 1], Q[i + 1], n);
    }
}