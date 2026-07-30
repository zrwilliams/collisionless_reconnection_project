"""
Dedalus script for calculating the maximum growth rates in no-slip
Rayleigh Benard convection over a range of horizontal wavenumbers.

This script can be ran serially or in parallel, and produces a plot of the
highest growth rate found for each horizontal wavenumber.

To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)


# Global parameters
Nx = 1024
ky_global = np.linspace(0.1, 0.9, 50)
Lx = np.pi
d_e = 0.04*2*Lx



# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
xcoord = d3.Coordinate('x') # Define x coordinate with simple Coordinate class as only working in 1d
dist = d3.Distributor(xcoord, dtype=np.complex128, comm=MPI.COMM_SELF) # Believe this is only used more if doing parallelization but needs to be called regardless for all problems
xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=3/2) # Now create actual basis for calculations to be done on

# Define dynamic variables psi and phi as d3 fields on the fourier basis
psi = dist.Field(name='psi',bases=xbasis)
phi = dist.Field(name='phi',bases=xbasis)
omega = dist.Field(name='omega')


## Substitutions ##
def max_growth_rate(ky):
    eta = 1.5e-3 # 3e-3
    rho_s = 3*d_e

    logger.info('Computing max growth rate for ky = %f' %ky)
    
    # Change ky parameter
    # Define derivative substitutions as lambda functions to be passed in to equation string through namespace 
    dx = lambda A: d3.Differentiate(A, xcoord)
    dy = lambda A: 1j*ky*A
    dt = lambda A: -1j*omega*A

    # Now define substitutions of laplcian variables again through lambda functions 
    Lap = lambda A: dx(dx(A)) + dy(dy(A))
    U = Lap(phi)
    J = - Lap(psi)
    F = psi + d_e*d_e*J

    # Define the non-constant coefficients as their own field objects 
    x = dist.local_grid(xbasis)
    dx_psi0 = dist.Field(bases=xbasis)
    dx_psi0['g'] = -np.sin(x)
    dx_J0 = dist.Field(bases=xbasis)
    dx_J0['g'] = -np.sin(x)
    dx_F0 = dist.Field(bases=xbasis)
    dx_F0['g'] = -(1+d_e*d_e)*np.sin(x)

    # Build the solver by passing variables and EV fields
    # Local namespace is also needed here to handle the previously defined constant variables into the equation string (ky,dx,J,etc.)
    problem = d3.EigenvalueProblem([psi,phi], eigenvalue=omega, namespace=locals())

    # Linearized equations can be entered with same rules as d2 (except for previous substitutions)
    # No need to define BCs (tau terms in d3) as Fourier basis assumes periodicity
    problem.add_equation("dt(U) + dx_psi0*dy(J) - dx_J0*dy(psi) = 0")
    problem.add_equation("dt(F) - dx_F0*dy(phi) + rho_s*rho_s*dx_psi0*dy(U) = 0")

    solver = problem.build_solver()

    logger.info('Computing max growth rate for ky = %f' %ky)
    
    # Change ky parameter 
    
    # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
    if (ky < 0.5):
        solver.solve_sparse(solver.subproblems[0], N=1, target=1j*0.1, rebuild_matrices=True)
    elif (ky < 0.8):
        solver.solve_sparse(solver.subproblems[0], N=1, target=1j*0.08, rebuild_matrices=True)
        # solver.solve_sparse(solver.subproblems[0], N=1, target=1j*1j*(1.-ky)*d_e, rebuild_matrices=True)
    else:
       solver.solve_sparse(solver.subproblems[0], N=1, target=1j*0.06, rebuild_matrices=True)
    # Return largest imaginary part
    return np.max(solver.eigenvalues.imag)

# Compute growth rate over local wavenumbers
ky_local = ky_global[CW.rank::CW.size]
t1 = time.time()
growth_local = np.array([max_growth_rate(ky) for ky in ky_local])
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Reduce growth rates to root process
growth_global = np.zeros_like(ky_global)
growth_global[CW.rank::CW.size] = growth_local
if CW.rank == 0:
    CW.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
else:
    CW.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

# Plot growth rates from root process
if CW.rank == 0:
    plt.plot(ky_global, growth_global, '.')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\mathrm{Im}(\omega)$')
    plt.title(f'Collisionless Tearing growth rates for d_e = {d_e:.4f}')
#    plt.savefig(f'/home/d3test/main/linear/{Nx}/growth_rate/d_e_{d_e:.2f}_growth_rates.png')
    plt.savefig(f'/home/test.png')


