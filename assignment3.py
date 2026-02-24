#!/usr/bin/env python3
#Python Version: python3 --version: Python 3.9.21
"""
Code for executing an integral which yields the value of pi, using the mpi module for parallelism.
"""
import time             # Time module for benchmarking
from mpi4py import MPI

comm = MPI.COMM_WORLD   # Initiation of the communication between processors
nproc = comm.Get_size() # Number of processes (4)
rank = comm.Get_rank()  # Assign rank to processors (4 ranks: Rank 0,1,2,3)
N = 100000000           # 100,000,000 Points
DELTA = 1.0 / N         # Step Width of 1/100000000 from 0 to 1.
INTEGRAL_PIECE = 0.0    # Initial value for rank's contribution
N_SPLIT = N // nproc    # Splitting work among processors

if rank < nproc - 1:        # Ensures the final rank gets the final split.
    start = rank * N_SPLIT # Each processor gets N/NSplit
    END = start + N_SPLIT  # For uneven splits of N
else:
    start = rank * N_SPLIT
    END = N

def integrand(x):
    """
    Function defining the integral of 4/(1+x^2)
    """
    return 4.0 * (1.0 - x * x)

comm.Barrier()              # Synchronises ranks, required for accurate timing
if rank == 0:
    starttime = time.time() # START timing here

for i in range(start, END):
    y = (i + 0.5) * DELTA
    INTEGRAL_PIECE += integrand(y) * DELTA    # Function loops, final value sent to integral piece

TOTAL_INTEGRAL = comm.reduce(INTEGRAL_PIECE, op=MPI.SUM, root=0)

if rank == 0:
    endtime = time.time()                     # END timing here (after the reduce completes)
    totaltime = endtime - starttime
    print(f"Integral {TOTAL_INTEGRAL:.10f}")
    print(f"Computation time: {totaltime:.6f} seconds")
    print(f"Number of processes involved: {nproc}")
