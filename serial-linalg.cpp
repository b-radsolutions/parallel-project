
#include "matrix-operations.hpp"

double serial_dot(double *a, double *b, size_t n) {
    double res = 0;
    for (size_t i = 0; i < n; i++) {
        res += a[i] * b[i];
    }
    return res;
}
