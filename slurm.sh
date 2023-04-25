#!/bin/bash -x

module load spectrum-mpi cuda/11.2
taskset -c 0-159:4 mpirun -N 4 /gpfs/u/home/PCPC/PCPCmchg/scratch/parallel-project/run.out
