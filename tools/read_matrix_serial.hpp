#pragma once
#ifndef MATRIX_WRITER_HPP
#define MATRIX_WRITER_HPP

#include <stdlib.h>
#include <string>

double **read_matrix(const std::string &filename, size_t n);
double **read_partial_matrix(const std::string &filename, size_t m, size_t n);

#endif
