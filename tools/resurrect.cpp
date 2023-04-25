
#include <fstream>
#include <stdio.h>
#include <string>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s [file prefix w/o extension] [number of parts] [output]\n",
               argv[0]);
        return 1;
    }

    size_t parts = atoi(argv[2]);

    std::string prefix = argv[1];
    std::string part = prefix + "0.mtx";

    // Open the first part and discover the size of matrix
    printf("Reading %s\n", part.c_str());
    std::ifstream file(part.c_str(), std::ios::in | std::ios::binary);
    if (!file) {
        printf("Failed to open %s\n", part.c_str());
        return 1;
    }
    size_t read_rows;
    file.read((char *)(&read_rows), sizeof(size_t));
    size_t vector_length = read_rows * parts;
    size_t expected_size = read_rows * vector_length * sizeof(double);
    char   buffer[expected_size];
    file.read(buffer, expected_size);
    file.close();

    size_t n = read_rows;

    // Open the output file
    std::ofstream output(argv[3], std::ios::out | std::ios::binary);
    if (!output) {
        printf("Failed to open output file %s\n", argv[3]);
        return 1;
    }
    // Write the output size
    output.write((char *)&vector_length, sizeof(size_t));
    // Write the first part
    output.write(buffer, expected_size);

    // Write the rest of the parts
    for (size_t i = 1; i < parts; i++) {
        std::string part = prefix + std::to_string(i) + ".mtx";
        printf("Reading %s\n", part.c_str());
        std::ifstream file(part.c_str(), std::ios::in | std::ios::binary);
        if (!file) {
            printf("Failed to open %s\n", part.c_str());
            return 1;
        }
        file.read((char *)(&read_rows), sizeof(size_t));
        if (read_rows != n) {
            printf("Part %s had incorrect size %lu (expected: %lu)\n", part.c_str(),
                   read_rows, n);
            return 1;
        }
        file.read(buffer, expected_size);
        output.write(buffer, expected_size);
        file.close();
    }
    output.close();

    printf("Wrote %lu parts to '%s', n=%lu\n", parts, argv[3], vector_length);
    return 0;
}
