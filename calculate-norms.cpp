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
    //printf("%s \n", filename.c_str());
    double **Q, **E;
    double frob_norm, one_norm, inf_norm;
    char bad = '.';
    if((filename.c_str())[0] == bad) {
        return;
    }
    Q = read_matrix(str1, m);
    E = orthoError(m, m, Q);
    frob_norm = frobeniusNorm(m, m, E);
    one_norm = oneNorm(m, m, E);
    inf_norm = infNorm(m, m, E);
    printf("%lu, %lf, %lf, %lf, %s\n", m, frob_norm, one_norm, inf_norm, filename.c_str());
}

int main(int argc, char* argv[]) {
    std::string filename = argv[1];
    std::ifstream input_file(filename);
    if (!input_file)
    {
        std::cerr << "Error: Could not open file \"" << filename << "\"" << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(input_file, line))
    {
        calculateError(line);
    }
    input_file.close();
    return 0;

}
