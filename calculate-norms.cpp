#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include "matrix-operations.hpp"
#include "orthogonality-test.hpp"
#include "tools/read_matrix_serial.hpp"
#define fold "./out/"

void calculateError(std::string filename) {
    char str1[100] = "";
    strcat(str1, fold);
    strcat(str1, filename.c_str());
    std::ifstream file(str1, std::ios::in | std::ios::binary);
    size_t m;
    file.read((char *)(&m), sizeof(size_t));

    double **Q, **E;
    double frob_norm, one_norm, inf_norm;

    Q = read_matrix(str1, m);
    E = orthoError(m, m, Q);
    frob_norm = frobeniusNorm(m, m, E);
    one_norm = oneNorm(m, m, E);
    inf_norm = infNorm(m, m, E);
    printf("%lu %lf %lf %lf %s\n", m, frob_norm, one_norm, inf_norm, filename.c_str());
}

int main(int argc, char* argv[]) {

     DIR* dir = opendir(fold);
    if (dir == NULL)
    {
        std::cerr << "Error: Could not open directory" << std::endl;
    }

    //printf("Size Frobenius One Inf\n");
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG)
        {
            calculateError(entry->d_name);
        }
    }
    closedir(dir);
}
