# parallel-project
The best group in Christopher C's. CSCI-4320 

Tasks:
- Classic Gram-Schmidt Process (in serial and parallel)
- Modified Gram-Schmidt Process (in serial and parallel)
    - Vector Projection
        - Dot Product
    - Vector Subtraction
    - Normalize
    - with storing computed dot products
- Load a matrix (using MPI Parallel I/O)
- Compute ||Q^T * Q - I||_2 (can be done serially, is not timed. Just for error calc)

Testing:
- Run on 1, 2, 4, 8, 16 compute nodes
- For sparse Matrices
- For dense Matrices
- Real data 
- Synthetic data
- Various matrix dimensions
- Well-conditioned and ill-conditioned matrices
- if time allows, gpu block size

PAPERS:

Parallel QR factorization
by Householder and modified
Gram-Schmidt algorithms
https://www.cs.umd.edu/users/oleary/reprints/j32.pdf

Efficient Parallel Implementation of Classical
Gram-Schmidt Orthogonalization Using Matrix
Multiplication
http://pmaa06.irisa.fr/papers_copy/41.pdf

Comparison of Different Parallel Modified Gram-Schmidt Algorithms
https://link.springer.com/chapter/10.1007/11549468_90

What do we know about
block Gram-Schmidt?
https://www2.karlin.mff.cuni.cz/~carson/ppt/ENLA_Carson.pdf

A Novel Parallel Algorithm Based on the Gram-Schmidt Method for Tridiagonal Linear Systems of Equations
https://www.hindawi.com/journals/mpe/2010/268093/

A Parallel implementation of Gram-Schmidt Algorithm for
Dense Linear System of Equations (NOTE:Take a look at their diagrams and their graphs)
https://www.ijcaonline.org/volume5/number7/pxc3871079.pdf

Analysis of Different Partition Schemes for parallel Gram-Schmidt Algorithms (NOTE: Acknowledge column parsing is faster when p>10)
http://homepage.cs.uiowa.edu/~oliveira/PAPERS/mgs-journal.pdf
