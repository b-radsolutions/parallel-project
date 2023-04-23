#pragma once
#ifndef GRAM_SCHMIDT_HPP
#define GRAM_SCHMIDT_HPP

void normal_gram_schmidt(double **A, size_t m, size_t n, double **Q);
void serial_modified_gram_schmidt(double **A, size_t m, size_t n, double **Q);
void parallel_modified_gram_schmidt(double **A, size_t m, size_t n, double **Q);
void parallel_gram_schmidt(double **A, size_t m, size_t n, double **Q);

#endif
