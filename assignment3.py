#!/usr/bin/env python3
#Python Version: python3 --version: Python 3.9.21
"""
Code for solving the Laplace and Poisson equations in 2D using different methods
"""
#ALL VALUES SHOULD BE TO 4 SIGNIFICANT FIGURES OF ACCURACY AT LEAST
# Task 1 RELAXATION SOLVER: Code a relaxation method to solve Poisson's equation for NxN grid, grid spacing h and specific charges at grid sites.
# This will be used to independently check our Monte Carlo results.
#===============================================================================================#

# Task 2 RANDOM WALK SOLVER: Code a random-walk solver to solve Poisson equation on the same grid, and obtain the Green's function value and its standard deviation
#===============================================================================================#


# Task 3: For a square grid of side length 1m, evaluate Green's function and it's error at
# a) Centre of grid (50cm, 50cm)
# b) Near corner (2cm, 2cm)
# c) Middle of a face? (2cm, 50cm)
# PLOT THE GREEN'S FUNCTION FOR ALL OF THESE AND ESTIMATE THE ERROR
#===============================================================================================#
import time
import math
import numpy as np
from mpi4py import MPI  # Defining over-relaxation function which loops over grid coordinates (i,j)

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

#====================================================#
# Defining Grid
#====================================================#
N = 100
L = 1          # 1 meter by 1 meter as per question 3                
h = L / (N-1)  # Grid = N x N = 100 x 100 = 10,000 points
               # Assuming the grid points touch the boundary, if N = 100, there are 99 gaps, then the spacing is the length of one side over number of gap.
omega = 2 / (1 + np.sin(np.pi / N))

# Defining charge functions f_ij

def charge_zero(i,j):
    return 0

def charge_uniform(i,j):
    return 10

#Calculated using gradient formula and plugged into y=mx+c
# j = 0, C = 0
# j = 1, C = N-1 [Since N begins at 0, not 1]
# y2-y1/x2-x1 = 1-0 / (N-1)-0 = 1/(N-1)
# y = mx+c or charge = mj + c where c = 0 since line crosses y axis
# charge = j/N-1
def charge_uniform_gradient(i,j):
    return j / (N-1)

# Distance formula from the centre.
# (0.5, 0.5) and (ih, jh)
def charge_exponential_decay(i,j):
    return np.exp(-10 * np.sqrt((i*h - 0.5)**2 + (j*h -0.5)**2))
    
#Defining phi with an empty array of size NxN, and applying boundary conditions
#Boundary Conditions: (: to select all of the dimension)
#                      (0 to select the first edge)
#                      (N-1 to select the last edge)

phi = np.zeroes([N,N]) # N rows and N columns (Rows is left to right, columns is up and down)
#If N=100, the rows go from 0 to 99, so the top row is index 99 = N-1
phiTop = phi[N-1, :] # [, :] is how to select all columns in numpy
phiBottom = phi[0, :] #N-1 selects only one corner, so use colon to get entire edge
phiLeft = phi[:, 0] #See notebook for deeper explanation
phiRight = phi[:, N-1]

#Defining f, the charge function which is comprised of interior charges
#f_ij * h^2 from the neighbours formula.
#Double loop to get an i and j line, and then the whole grid.
#Think about two perpindicular lines combining to make an entire square.
f[i,j] = 

#i and j are undefined, from N = 1 to 100, or in terms of python arrays: 1 - N-1
#
for i in range(1, N-1):
    for j in range(1, N-1):
        phi[i,j] = omega * (f[i,j] + 0.25*(phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])) + (1-omega) * phi[i,j]
