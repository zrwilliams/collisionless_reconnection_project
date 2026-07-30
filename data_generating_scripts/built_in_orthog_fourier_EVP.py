'''
Nathan's script, with sparing edits by Dr. Williams and Isaac
'''

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import dedalus.public as d3
import time
import logging
import h5py
import scipy.linalg
logger = logging.getLogger(__name__)
import scipy

# Function only needed for calculating energy of the system
def compute_energies(solver,phi,psi):
    # Do not need to integrate in y direction as we have already chosen a specific ky
    # Construct energy operator
    print("Now solving for energy")
    xcoord2 = d3.Coordinate('x') # Define x coordinate with simple Coordinate class as only working in 1d
    dist2 = d3.Distributor(xcoord2, dtype=np.complex128) # Believe this is only used more if doing parallelization but needs to be called regardless for all problems
    xbasis2 = d3.ComplexFourier(xcoord2, Nx, bounds=(-Lx, Lx),dealias=1) 

    psix = d3.Differentiate(psi,xcoord2)
    psixx = d3.Differentiate(psix,xcoord2)
    phix = d3.Differentiate(phi,xcoord2)

    pre_J_op = d_e**2 * (psixx + ky**2*psi)**2
    
    pre_G_psi_op = (psix*np.conj(psix) + ky*psi*np.conj(psi))
    pre_G_phi_op = (phix*np.conj(phix)+ ky*phi*np.conj(phi))

    J_op = 1/2 * d3.Integrate(pre_J_op,'x')
    G_psi_op = 1/2 * d3.Integrate(pre_G_psi_op,'x')
    G_phi_op = 1/2 * d3.Integrate(pre_G_phi_op,'x')

    
    E_op = J_op + G_psi_op + G_phi_op
    
    # Evaluate energy for each mode
    N = Nx # len(solver.eigenvalues)
    energies = np.zeros(N)
    G_phi_energies = np.zeros(N)
    G_psi_energies = np.zeros(N)
    J_energies = np.zeros(N)

    for i in range(N):
        solver.set_state(i,solver.subsystems[0])
        G_phi_energies[i] = np.abs(G_phi_op.evaluate()['c'][0])
        G_psi_energies[i] = np.abs(G_psi_op.evaluate()['c'][0])
        J_energies[i] = np.abs(J_op.evaluate()['c'][0])
        energies[i] = np.abs(E_op.evaluate()['c'][0])

    
    return energies, G_phi_energies, G_psi_energies, J_energies


## Parameters ##

# Input parameters needed for calculation
Lx = np.pi
Nx = 256
ky = 0.5
d_e = 0.04*2*Lx
normalize_rights = False # For some reason this scales the x dimensionality by the dealias factor even though that shouldnt change
run_num = '209'


## Setup ##

# Set up dedalus3 Fourier basis
xcoord = d3.Coordinate('x') # Define x coordinate with simple Coordinate class as only working in 1d
dist = d3.Distributor(xcoord, dtype=np.complex128) # Believe this is only used more if doing parallelization but needs to be called regardless for all problems
xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=3/2) # Now create actual basis for calculations to be done on

# Define dynamic variables psi and phi as d3 fields on the fourier basis
psi = dist.Field(name='psi',bases=xbasis)
phi = dist.Field(name='phi',bases=xbasis)
# Eigenvalues also need to be defined as a field for the solver but do not need to have a basis 
omega = dist.Field(name='omega')


## Substitutions ##

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
problem.add_equation("dt(F) - dx_F0*dy(phi) = 0")


## Rest of the script is very similar to d2 EVP solver minus some slight changes to the names of some operations 

# Build Solver
t1 = time.time()
solver = problem.build_solver()
solver.solve_dense(solver.subproblems[0], left=True, normalize_left=True)

print(f"shape is {np.shape(np.transpose(solver.modified_left_eigenvectors).conj())}")

#### Orthog based on solver.eigenvectors ####
print(f"shape is {np.shape(solver.right_eigenvectors)}")
kronecker = np.zeros((len(solver.modified_left_eigenvectors[:,0]),len(solver.right_eigenvectors[:,0])))
for i in range(len(solver.modified_left_eigenvectors[:,0])):
    for j in range(len(solver.right_eigenvectors[:,0])):
        kronecker[i,j] = np.transpose(solver.modified_left_eigenvectors).conj()[i]@np.transpose(solver.right_eigenvectors)[j]

# print(np.shape(kronecker))

#### Orthog based on Fields ####
for n in range(len(solver.eigenvalues)):    
    if n==0:
        phi_r = np.zeros((len(solver.eigenvalues), len(phi['c'])), dtype=np.complex128)
        psi_r = np.zeros_like(phi_r)
        phi_l = np.zeros_like(phi_r)
        psi_l = np.zeros_like(phi_r)

    solver.set_state(n, solver.subsystems[0])
    phi_r[n][:] = phi['c']
    psi_r[n][:] = psi['c']

    solver.set_state(n, solver.subsystems[0], LEFT=True)
    phi_l[n][:] = phi['c']
    psi_l[n][:] = psi['c']

kronecker2 = np.zeros((len(solver.modified_left_eigenvectors[:,0]),len(solver.right_eigenvectors[:,0])))

for i in range(len(solver.modified_left_eigenvectors[:,0])):
    for j in range(len(solver.right_eigenvectors[:,0])):
        kronecker2[i,j] = np.transpose(phi_l[i]).conj()@np.transpose(phi_r[j]) + np.transpose(psi_l[i]).conj()@np.transpose(psi_r[j])



# plt.matshow(np.abs(kronecker))
plt.matshow(np.abs(kronecker),norm=colors.LogNorm())
plt.colorbar()
plt.title(f'Orthogonality Check Linear {run_num} ')
plt.show()
plt.savefig('test.png',format='png',dpi=600)
#plt.savefig(f'/home/d3test/main/linear/orthog/dot/field_test_orthogonality.png',format='png',dpi=150,bbox_inches='tight')     
plt.close()


# if LEFT:
#     #filename = f'/home/d3test/main/linear/{Nx+1}/eigenmodes/{run_num}_{Nx+1}_left.h5'
#     filename = f'/home/d3test/test_left.h5'
# else:
#     #filename = f'/home/d3test/main/linear/{Nx+1}/eigenmodes/{run_num}_{Nx+1}_right.h5'
#     filename = f'/home/d3test/test_right.h5'

# logger.info('Saving Output to H5PY File:'+str(filename)+'\n')
# hfile = h5py.File(filename,'w')

# for n in range(len(solver.eigenvalues)):
#     solver.set_state(n,solver.subsystems[0])

#     if LEFT:
#         solver.set_state(n, solver.subsystems[0], LEFT=True)
#     else:
#         solver.set_state(n, solver.subsystems[0])
#     if n==0:
#         phi_s = np.zeros((len(solver.eigenvalues), len(phi['c'])), dtype=np.complex128)
#         psi_s = np.zeros_like(phi_s)

#     phi_s[n][:] = phi['c']
#     psi_s[n][:] = psi['c']

# print(f"psi shape is {np.shape(psi_s)}")
# tasks = hfile.create_group('tasks')

# tasks.create_dataset('phi', data=phi_s)
# tasks.create_dataset('psi', data=psi_s)

# scales = hfile.create_group('scales')
# scales.create_dataset('growth_rate',data=solver.eigenvalues.imag)
# scales.create_dataset('frequency',data=solver.eigenvalues.real)
# scales.create_dataset('complex_EV',data=solver.eigenvalues)

# if not LEFT:
#     energies = hfile.create_group('energies')
#     energies.create_dataset('total_energy', data=energies)


# hfile.close()

# t2 = time.time()
# logger.info('Saving completed, elapsed solve time: %f'%(t2-t1))