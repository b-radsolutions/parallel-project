#pragma once
#ifndef MATRIX_WRITER_HPP
#define MATRIX_WRITER_HPP

#include <stdlib.h>
#include <string>

int write_matrix_to_file_serial(double **A, size_t n, const std::string& filename);

#endif
