
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
    size_t read_n;
    file.read((char *)(&read_n), sizeof(size_t));
    printf("%lu\n", read_n);
    size_t expected_size = read_n * read_n / parts * sizeof(double);
    char   buffer[expected_size];
    file.read(buffer, expected_size);
    file.close();

    size_t n = read_n;

    // Open the output file
    std::ofstream output(argv[3], std::ios::out | std::ios::binary);
    if (!output) {
        printf("Failed to open output file %s\n", argv[3]);
        return 1;
    }
    // Write the size
    output.write((char *)&n, sizeof(size_t));
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
        file.read((char *)(&read_n), sizeof(size_t));
        if (read_n != n) {
            printf("Part %s had incorrect size %lu (expected: %lu)\n", part.c_str(),
                   read_n, n);
            return 1;
        }
        file.read(buffer, expected_size);
        output.write(buffer, expected_size);
        file.close();
    }
    output.close();

    printf("Wrote %lu parts to '%s', n=%lu\n", parts, argv[3], n);
    return 0;
}
