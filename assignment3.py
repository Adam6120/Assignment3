#!/usr/bin/env python3
# ©(2026) ADAM HUSSAIN. ALL RIGHTS RESERVED
#Python Version: python3 --version: Python 3.9.21
"""
Code for solving the Laplace and Poisson equations in 2D using different methods
"""
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI  # Defining over-relaxation function which loops over grid coordinates (i,j)

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()
N_WALKERS = 1000000   # Number of Walkers
N_SPLIT = N_WALKERS // nproc   # Splitting work among processors
results = [] #Empty array to catch the outputs before they go into the table

#====================================================#
# Defining Grid
#====================================================#
N = 100
L = 1          # 1 meter by 1 meter as per question 3           
h = L / (N-1)  # Grid = N x N = 100 x 100 = 10,000 points
               # Assuming the grid points touch the boundary, if N = 100, there are 99 gaps, then
               # the spacing is the length of one side over number of gap.
omega = 2 / (1 + np.sin(np.pi / N))

# Defining charge functions f_ij

def charge_zero(i,j):
    """
    Returns zero charge. 
    """
    return 0

def charge_uniform(i,j):
    """
    Returns uniform charge of 10.
    """
    return 10

#Calculated using gradient formula and plugged into y=mx+c
# j = 0, C = 0
# j = 1, C = N-1 [Since N begins at 0, not 1]
# y2-y1/x2-x1 = 1-0 / (N-1)-0 = 1/(N-1)
# y = mx+c or charge = mj + c where c = 0 since line crosses y axis
# charge = j/N-1
def charge_uniform_gradient(i,j):
    """
    Returns uniform gradient from bottom of grid to top.
    """
    return j / (N-1)

# Distance formula from the centre.
# (0.5, 0.5) and (ih, jh)
def charge_exponential_decay(i,j):
    """
    Return an exponentially decaying charge distribution centred at (0.5, 0.5)."
    """
    return np.exp(-10 * np.sqrt((i*h - 0.5)**2 + (j*h -0.5)**2))



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
        #f[i,j] = charge_function[i,j] * (h**2)

#ChargeFunction parameter whihc will take task 3 charge values.
def relaxation_solver(charge_function, top, bottom, left, right):
    """
    Over-relaxation method for the Poisson
    equation
    """

    #Potential boundaries
    phi = np.zeros([N,N])
    phi[N-1, :] = top
    phi[0, :] = bottom
    phi[:, 0] = left
    phi[:, N-1] = right

    #Charge
    f = np.zeros([N,N])
    for i in range(1, N-1):
        for j in range(1, N-1):
            f[i,j] = charge_function(i,j) * (h**2)

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
    visits = [] # Records the path of walkers BEFORE hitting the boundary


    while True:
        #Check if boundary has been hit
        if i == 0 or i == N-1 or j == 0 or j == N-1:
            return i, j, visits

        visits.append((i,j)) #This just adds the coordinates into the visits array
        #If we consider a random walker at the 2D point (i, j) and allow it to jump
        #in one of four directions
        #with equal probability (i.e. 0.25) and we average the potential found by each walker.
        #Assigning a random integer to one direction, which walker will travel
        step = random.randint(0,3)
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

def greens_function(start_i, start_j):
    """
    Greens function for a starting point (i, j)
    using random walkers
    """
    count_piece = np.zeros((N,N)) # Empty array for ranks to place their boundary counts inside.
    visit_piece = np.zeros((N,N)) # Empty array for ranks to place their paths inside

    for _ in range(start, end): #Changed from for walker in range to for _ in range
        boundary_i, boundary_j, visits = random_walker(start_i, start_j)
        count_piece[boundary_i, boundary_j] += 1 # Add one count to count piece
        for (visit_i, visit_j) in visits:
            visit_piece[visit_i, visit_j] += 1 #Add one visit to visit piece


    count_global = np.zeros((N,N)) # All ranks contribute to this
    visit_global = np.zeros((N,N)) # All ranks contribute to this
    #Comm.Reduce(sendbuf, recvbuf, op=operation, root=0)
    comm.Reduce(count_piece, count_global, MPI.SUM, root=0)     # Summing all rank count pieces
    comm.Reduce(visit_piece, visit_global, MPI.SUM, root=0)     # Summing all rank visit pieces

    # Example STD DEV Calculation
    # For one boundary point (i,j), walkers landed there? yes = 900, no = 100
    # p = 900/1000 = 0.9, Variance = p(1-p) = 0.9(0.1) = 0.09
    # Variance of 1000 summed trials = 1000 * 0.09 = 90
    # Interested in the variance of the average probability of landing at (i,j)
    # 90/1000^2 = 0.00009, STD DEV = sqrt(0.00009) = 0.0095
    # So probability of 0.9 has an uncertainty of +- 0.0095
    # STD DEV = sqrt(p(1-p) / N_WALKERS)

    landing_probability = count_global / N_WALKERS #Probabilities
    std_dev = np.sqrt(landing_probability*(1-landing_probability) / N_WALKERS)
    charge_greens = (h**2 / N_WALKERS) * visit_global
    return landing_probability, std_dev, charge_greens

# Task 3 Evaluation ===============================
# Converting coordinates into grid points (ih, jh), h = L/(N-1) = 1/99
# i) (50cm, 50cm) = (0.5m, 0.5m), thus ih, jh = 0.5 => i, j = 0.5 / h = 99 * 0.5 = 49.5 = 50
# (rounded)
# i) = (50,50)
# ii) = (2,2)
# iii) = (2,50)
points = [(50, 50), (2,2), (2,50)]

#Defining Boundary Conditions BC a), b), c) = top, bottom, left, right
BC_A = (100, 100, 100, 100) #All edged uniform +100V
BC_B = (100, 100, -100, -100)
BC_C = (200, 0, 200, -400)

charge_types = [charge_zero, charge_uniform, charge_uniform_gradient, charge_exponential_decay]


for (start_i, start_j) in points:
    comm.Barrier() #Synchronising Ranks For Accurate Timing
    if rank == 0:
        starttime = time.time() #Starting timer once all ranks are ready

    landing_probability, std_dev, charge_greens = greens_function(start_i, start_j)


    comm.Barrier()
    if rank == 0:
        endtime = time.time()
        print(f"Number of processors: {nproc}")
        print(f"Timing: {endtime - starttime:.4f} seconds")

        # Plotting random walker landing probability
        # Colour only shows on the edges, expected the entire grid to be full
        plt.imshow(landing_probability, origin='lower', cmap='rainbow')
        plt.title(f'Greens Function at ({start_i},{start_j})')
        plt.colorbar()
        plt.savefig(f'greens_{start_i}_{start_j}_{nproc}.png', dpi=300)
        plt.close()

        # Plotting random walker interior paths
        plt.imshow(charge_greens, origin='lower', cmap='rainbow')
        plt.title(f'Charge Greens Function at ({start_i},{start_j}), Walkers:{N_WALKERS}, Processors:{nproc}')
        plt.colorbar()
        #Trying to get better resolution as plots seem blurry, seems dpi just increases the size
        plt.savefig(f'charge_greens_{start_i}_{start_j}_{nproc}.png', dpi=300)
        plt.close()

        for (top, bottom, left, right) in BC_A, BC_B, BC_C:
            #Unpacking boundary condition tuples into values of voltages for each edge
            for charge_function in charge_types:
                f_grid = np.zeros((N,N)) #Filling this grid with charge values across grid
                for i in range(1, N-1):
                    for j in range(1, N-1):
                        f_grid[i,j] = charge_function(i,j)
                potential = (
                    np.sum(landing_probability[N-1, :] * top
                          + landing_probability[0, :] * bottom    # pylint flagged > 100 char
                          + landing_probability[:, 0] * left
                          + landing_probability[:, N-1] * right)
                    + np.sum(charge_greens * f_grid)
                )

                relaxation = relaxation_solver(charge_function, top, bottom, left, right)
                results.append(
                    (
                        start_i,
                        start_j,
                        top,
                        bottom,
                        left,
                        right,
                        charge_function.__name__,
                        potential,
                        relaxation[start_i, start_j]
                    )
                )
if rank == 0:
    print(f"{'Point':<12} {'BC':<20} {'Charge':<30} {'Greens (V)':<15} {'Relaxation (V)':<15}")
    print("-" * 92)
    for (start_i, start_j, top, bottom, left, right, charge_name, potential, relax_potential) in results:
        print(f"({start_i},{start_j}){'': <8} ({top},{bottom},{left},{right}){'': <4} {charge_name:<30} {potential:<15.4f} {relax_potential:<15.4f}") #Table formatting
