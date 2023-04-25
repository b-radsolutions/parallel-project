#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <dirent.h>
#include "matrix-operations.hpp"
#include "orthogonality-test.hpp"
#include "tools/read_matrix_serial.hpp"

void calculateError(std::string filename, std::string fold) {
    char str1[100] = "";
    strcat(str1, fold.c_str());
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

//either accepts a directory if flag -r is provided before the name of the directory
//or accepts a single file
int main(int argc, char *argv[])
{
    printf("%s\n", argv[1]);
    if(strcmp(argv[1], "-r") == 0) {
        printf("%s\n", argv[2]);
        std::string folder = argv[2];
        DIR *dir = opendir(folder.c_str());
        if (dir == NULL)
        {
            std::cerr << "Error: Could not open directory" << std::endl;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL)
        {
            if (entry->d_type == DT_REG)
            {
                calculateError(entry->d_name, folder);
            }
        }
        closedir(dir);
    }
    else {
        calculateError(argv[1], "");
    }
    return 0;
}
