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

#Notes for tracking my understanding:
# we are defining everything we've been doing recently into the sor solver function,
#we're initialising phi with our boundary conditions, but not assigning actual conditions yet.
#Then we define f which is our charge function initially as a bunch of zeros for a grid of NxN
#we do a double loop to cover the entire grid with two lines from our i and j loop, then
#we finally define f[i,j] as f_ij h^2 which is taken from our 1/4 neighbours equation for phi_i,j]
#next we start the convergence malarkey which is essentially means everything becomes zero,
#in this case, do all the values of phi for each point i and j equal zero?
#this is dependent on the 4 neighbours of each point reaching their final values.
#once they have, we can safely assume the walkers have covered the entire grid and we can end the loop

import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI  # Defining over-relaxation function which loops over grid coordinates (i,j)

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()
N_WALKERS = 1000000    # Number of Walkers
N_SPLIT = N_WALKERS // nproc   # Splitting work among processors

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

phi = np.zeros([N,N]) # N rows and N columns (Rows is left to right, columns is up and down)
#If N=100, the rows go from 0 to 99, so the top row is index 99 = N-1
phiTop = phi[N-1, :] # [, :] is how to select all columns in numpy
phiBottom = phi[0, :] #N-1 selects only one corner, so use colon to get entire edge
phiLeft = phi[:, 0] #See notebook for deeper explanation
phiRight = phi[:, N-1]

#Defining f, the charge function which is comprised of interior charges
#f_ij * h^2 from the neighbours formula.
#Double loop to get an i and j line, and then the whole grid.
#Think about two perpindicular lines combining to make an entire square.
f = np.zeros([N,N])
#for i in range(1, N-1):
    #for j in range(1, N-1):
        #f[i,j] = chargeFunction[i,j] * (h**2)
        
#ChargeFunction parameter whihc will take task 3 charge values.
def RelaxationSolver(chargeFunction, top, bottom, left, right):
    """
    Over-relaxation method for the Poisson
    equation 
    """
    
    #Potential boundaries
    phi = np.zeros([N,N])
    phiTop = phi[N-1, :] = top #Forgot to actually define these empty arrays
    phiBottom = phi[0, :] = bottom
    phiLeft = phi[:, 0] = left
    phiRight = phi[:, N-1] = right
    
    #Charge
    f = np.zeros([N,N])
    for i in range(1, N-1):
        for j in range(1, N-1):
            f[i,j] = chargeFunction(i,j) * (h**2)
            
    #Ensuring phi returns after convergence has been reached
    while True: #The loop will continue until told to stop
        max_potential_difference = 0 #Value of potential at point (i,j)
        for i in range(1, N-1): #All rows
            for j in range(1, N-1): #All columns
                x = phi[i,j] #Point value becomes x, this is how change is calculated
                phi[i,j] = omega * (f[i,j] + 0.25*(phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])) + (1-omega) * phi[i,j]
                potential_difference = abs(phi[i,j] - x)
                if potential_difference > max_potential_difference:
                    max_potential_difference = potential_difference
                    
        if max_potential_difference < 0.0001: #tolerance
            break
    return phi
#i and j are undefined, from N = 1 to 100, or in terms of python arrays: 1 - N-1
#
for i in range(1, N-1):
    for j in range(1, N-1):
        phi[i,j] = omega * (f[i,j] + 0.25*(phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])) + (1-omega) * phi[i,j]
        
        
#Task 2 - Random Walk solver ===================================
def random_walker(initial_i, initial_j): #initial starting positions (i,j)
    """
    Single random walker starting at (initial_i, initial_j).
    Walks randomly until it hits the boundary.
    Returns the boundary point it landed on.
    """

    i, j = initial_i, initial_j
    
    
    while True:
        #Check if boundary has been hit
        if i == 0 or i == N-1 or j == 0 or j == N-1:
            return i, j
        
        #If we consider a random walker at the 2D point (i, j) and allow it to jump in one of four directions
        #with equal probability (i.e. 0.25) and we average the potential found by each walker.
        step = random.randint(0,3) #Assigning a random integer to one direction, which walker will travel
        if step == 0: #(i,j) = (x,y), up = j+1, right = i+1.. etc
            j = j + 1 # Up
        
        elif step == 1:
            j = j - 1 # Down
            
        elif step == 2:
            i = i + 1 # Right
            
        else:
            i = i - 1 # Left
            
        
# Task 2 ===========================================
#Implement a random-walk solver for the Poisson equation on a similar grid,
#to obtain the Green's function. This code should start all of the walkers at a
#specified grid point inside the two dimensional region and walk them to the surrounding boundary
#surface. Calculate both the value of the Green's function and its standard deviation.

if rank < nproc - 1:                # Ensures the final rank gets the final split.
    start = rank * N_SPLIT          # Each processor gets N/NSplit
    end = start + N_SPLIT           # For uneven splits of N
else:
    start = rank * N_SPLIT
    end = N_WALKERS
    
def greensFunction(start_i, start_j):
    """
    Greens function for a starting point (i, j)
    using random walkers
    """
    COUNT_PIECE = np.zeros((N,N)) # Empty array for ranks to place their boundary counts inside.
        
    for walker in range(start, end):
        boundary_i, boundary_j = random_walker(start_i, start_j)
        COUNT_PIECE[boundary_i, boundary_j] += 1 # Add one count to count piece
        COUNT_GLOBAL = np.zeros((N,N))    
        #Comm.Reduce(sendbuf, recvbuf, op=operation, root=0)
        comm.Reduce(COUNT_PIECE, COUNT_GLOBAL, MPI.SUM, root=0)     # Summing all rank count pieces

        
        # Example STD DEV Calculation
        # For one boundary point (i,j), walkers landed there? yes = 900, no = 100
        # p = 900/1000 = 0.9, Variance = p(1-p) = 0.9(0.1) = 0.09
        # Variance of 1000 summed trials = 1000 * 0.09 = 90
        # Interested in the variance of the average probability of landing at (i,j)
        # 90/1000^2 = 0.00009, STD DEV = sqrt(0.00009) = 0.0095
        # So probability of 0.9 has an uncertainty of +- 0.0095
        # STD DEV = sqrt(p(1-p) / N_WALKERS)
        
        landing_probability = COUNT_GLOBAL / N_WALKERS #Probabilities
        std_dev = np.sqrt(landing_probability*(1-landing_probability) / N_WALKERS)
        return landing_probability, std_dev
    
# Task 3 Evaluation ===============================
# Converting coordinates into grid points (ih, jh), h = L/(N-1) = 1/99
# i) (50cm, 50cm) = (0.5m, 0.5m), thus ih, jh = 0.5 => i, j = 0.5 / h = 99 * 0.5 = 49.5 = 50 (rounded)
# i) = (50,50)
# ii) = (2,2)
# iii) = (2,50)
points = [(50, 50), (2,2), (2,50)]

#Defining Boundary Conditions BC a), b), c) = top, bottom, left, right
BC_A = (100, 100, 100, 100) #All edged uniform +100V
BC_B = (100, 100, -100, -100)
BC_C = (200, 0, 200, -400)

charge_types = [charge_zero, charge_uniform, charge_uniform_gradient, charge_exponential_decay]

comm.Barrier() #Synchronising Ranks
if rank == 0:
    starttime = time.time() #Starting timer 
    
for (start_i, start_j) in points: 
    landing_probability, std_dev = greensFunction(start_i, start_j)
    
    if rank == 0:
        for (top, bottom, left, right) in BC_A, BC_B, BC_C:
            #Unpacking boundary condition tuples into values of voltages for each edge
            for chargeFunction in charge_types:
                GTop = landing_probability[N-1, :]
                GBottom = landing_probability[0, :]
                GLeft = landing_probability[:, 0]
                GRight = landing_probability[:, N-1]
                
                potential = np.sum((GTop * top ) + (GBottom * bottom) + (GLeft * left) + (GRight * right))
                print(f"Potential: {potential:.4f} V") #Printing Voltage to 4 sig figs
                
                relaxation = RelaxationSolver(chargeFunction, top, bottom, left, right)
                print(f"relaxation: {relaxation[start_i, start_j]:.4f} V") #Printing Relaxation to 4 sig figs
                
#Greens Function for calculating potential i_j
#G(i, j, xb, yb) * phi(xb, yb) is the probability of a walker
#starting at point (i,j) and landing at boundary (xb, yb)
#phi(xb, yb) is the voltage at (xb, yb)

#Numbered example for clarity, 2000 walkers from (0.5, 0.5)
# How many walkers landed on the boundaries?
# Top: 1800, p = 1800/2000 = 0.9               This is G(top)
# Bottom: 0, p = 0                                     G(bottom)
# Left: -100, p = 0.05                                  G(left)
# Right: -100, p = 0.05                                 G(right)
#Applying random Boundary Conditions, top and bottom +100V, left and right -100V
# phi(0.5, 0.5) = G(top) * 100   + G(bottom) * 100 + G(left) * -100 + G(right) * -100   
# = 0.9 * 100 + 0 - 0.05 * 100 - 0.05*100 = 80V
# Thus the point (0.5, 0.5) feels 80V, similar to astronomical stabilisation of gravity (see notes)

#So we need to take the walker from point (i,j) and take the sum of the landing probability
# multiplied by the boundary potentials of all the edges.

# Second term is the charge contribution

# Using matplotlib to plot the NXN grid greens function for our test points

