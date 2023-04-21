
#include <fstream>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s [name of output file] [matrix size 'N']\n", argv[0]);
        return 1;
    }
    // Get the filename to write to
    char *filename = argv[1];
    // Get the number N of the matrix
    size_t n = std::atoi(argv[2]);

    printf("Generating matrix '%s' with size %lux%lu\n", filename, n, n);

    // Open the file
    std::ofstream wf(filename, std::ios::out | std::ios::binary);
    if (!wf) {
        printf("Failed to open file '%s'\n", filename);
        return 1;
    }

    // The first value we insert will be the size of the matrix as a size_t
    wf.write((char *)&n, sizeof(size_t));

    // Create and insert double values
    size_t number_elements = n * n;
    for (size_t i = 0; i < number_elements; i++) {
        double value = ((double)rand() / (double)RAND_MAX);
        wf.write((char *)&value, sizeof(double));
    }
    wf.close();
    printf("Generated %lu elements.\nDone.\n", number_elements);
}
