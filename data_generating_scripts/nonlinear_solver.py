import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import h5py
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

# Equation Parameters  
Ny, Nx = 32, 511 #64, 1023 # Resolutions are best used in powers of 2, x-resolution (Nx) chosen to be 1 less than power of 2 to remove nyquist mode from linear eigenmode solutions
Ly, Lx = 2*np.pi, np.pi 
d_e = 0.04*2*Lx # electron inertial effects
eta = 0 #1.5e-3 # resistive effects
mu_e = 0 #1e-2 # dissipation in phi
mu_psi = 0 #1e-2 # dissipation in psi

# Script Parameters
init_dt = 125e-6
stop_sim_time = np.inf
run_num = '999'
dissipation = False
resistivity = False
timestepper = d3.RK443

# Optional initial conditions
# 'default' - gaussian in x, cos in y
# 'eigen' - reads in linear sol data and sets psi/phi data to most unstable eigenmode
# 'rand' - fill grids with random noise, imaginary part has to be zeroed out otherwise simulation will immediately crash  
initial_condition = 'default'



# Create bases and domain
coords = d3.CartesianCoordinates('y','x')
dist = d3.Distributor(coords, dtype=np.complex128)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly, Ly), dealias=3/2)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx, Lx), dealias=3/2)
y, x = dist.local_grids(ybasis, xbasis)

# Dynamic fields, single tau term needed
psi = dist.Field(name='psi', bases=(ybasis,xbasis))
phi = dist.Field(name='phi', bases=(ybasis,xbasis))
tau_phi = dist.Field(name='tau_phi') # tau terms 

# Define derivative substitutions as lambda functions to be passed in to equation string through namespace 
dy = lambda A: d3.Differentiate(A, coords['y'])
dx = lambda A: d3.Differentiate(A, coords['x'])

# Now define substitutions of laplcian variables again through lambda functions 
Lap = lambda A: dx(dx(A)) + dy(dy(A))
U = Lap(phi)
J = - Lap(psi)
F = psi + d_e*d_e*J

# Define the non-constant coefficients as their own field objects - only needed in xbasis 
dx_psi0 = dist.Field(bases=(xbasis))
dx_psi0['g'] = -np.sin(x)
J0 = dist.Field(bases=(xbasis))
J0['g'] =  np.cos(x)
dx_J0 = dist.Field(bases=(xbasis))
dx_J0['g'] =  -np.sin(x)
dx_F0 = dist.Field(bases=(xbasis))
dx_F0['g'] =  -(1+d_e*d_e)*np.sin(x)

# CFL velocities - for timestep 
ey, ex = coords.unit_vector_fields(dist)
v = dist.VectorField(coords, name='v', bases=(ybasis,xbasis))
v = - dy(phi)*ex + dx(phi)*ey 

B = dist.VectorField(coords, name='B', bases=(ybasis,xbasis))
B = dy(psi)*ex - dx(psi)*ey


# Problem statement
problem = d3.InitialValueProblem(variables=[phi, psi, tau_phi], namespace=locals())

# Actual equations - Important to remember that LHS is implicit and RHS is explicit. Nonlinear terms on RHS 
# Equations with dissapation
if dissipation:
    problem.add_equation("dt(U) - dx_J0*dy(psi) + dx_psi0*dy(J) - mu_e*Lap(U) + tau_phi = -dx(phi)*dy(U) + dx(U)*dy(phi) + dx(J)*dy(psi) - dx(psi)*dy(J)")
    if resistivity:
        problem.add_equation("dt(F) - dx_F0*dy(phi) - mu_psi*Lap(J) + eta*J = dx(F)*dy(phi) - dx(phi)*dy(F) + mu_psi*Lap(J0)")
    else:
        problem.add_equation("dt(F) - dx_F0*dy(phi) - mu_psi*Lap(J)  = dx(F)*dy(phi) - dx(phi)*dy(F) + mu_psi*Lap(J0)")

# Equations without dissipation
else:
    problem.add_equation("dt(U) - dx_J0*dy(psi) + dx_psi0*dy(J) + tau_phi = -dx(phi)*dy(U) + dx(U)*dy(phi) + dx(J)*dy(psi) - dx(psi)*dy(J)")
    if resistivity:
            problem.add_equation("dt(F) - dx_F0*dy(phi) + eta*J = dx(F)*dy(phi) - dx(phi)*dy(F) + mu_psi*Lap(J0)")
    else:
            problem.add_equation("dt(F) - dx_F0*dy(phi) = dx(F)*dy(phi) - dx(phi)*dy(F)")


problem.add_equation("integ(phi) = 0") # Gauge condition

# Build solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
logger.info('Solver built')

# Initial conditions
if initial_condition == 'eigen':
    # Read in eigenmode data
    lin_run = 221
    kyi = 1
    amp = 1e-4
    Right_Eigen_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx+1}/eigenmodes/{lin_run}_{Nx+1}_right.h5'
    with h5py.File(Right_Eigen_Path,mode='r') as file:
        eigen_psi =  np.array(file['tasks/psi'])
        eigen_phi = np.array(file['tasks/phi'])
        gammas = np.array(file['scales/growth_rate'])

    # Find unstable mode & set initial conditions
    unstable_index = np.argmax(gammas)
    psi['c'][kyi,:] = amp*eigen_psi[unstable_index,:]
    phi['c'][kyi,:] = amp*eigen_phi[unstable_index,:]

elif initial_condition == 'rand':
    amp = 1e-4
    sigma = 2.0
    psi.fill_random('g', seed=42, distribution='normal', scale=1e-5)
    phi.fill_random('g', seed=36, distribution='normal', scale=1e-5)

    for i in range(6,64):
        psi['c'][i,:] = 0
        phi['c'][i,:] = 0

else:
    mixlayout = dist.layouts[2]
    amp = 1e-4 # 0.0005 
    sigma = 2.0
    psi['g'] = amp*np.exp(-x**2/sigma**2)*np.cos((Lx/Ly)*y) # used for paper 
    # psi['g'] = amp*np.exp(-x**2/sigma**2)*np.exp(-y**2/sigma**2) # double gaussian



# Analysis 
fh_mode = 'overwrite'
savepath = f'/media/williams/plasma_data/collisionless_tearing_project/2024/nonlinear/{Nx+1}/snapshots/{run_num}_nonlinear_sim'
snapshots = solver.evaluator.add_file_handler(savepath, sim_dt=1, max_writes=5000, mode=fh_mode) 

snapshots.add_task(phi, layout='g', name='phi')
snapshots.add_task(psi, layout='g', name='psi')

# CFL - updates step size (dt) based off of current conditions
CFL = d3.CFL(solver,initial_dt=init_dt,cadence=1,safety=0.8,max_change=1.5,max_dt=1)
CFL.add_velocity(v) # phi stuff 
CFL.add_velocity(B) # psi stuff

# Flow properties - gives updates durring simulation 
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(v@v), name='v') #phi stuff # @ symbol useful for computing dot products
flow.add_property(np.sqrt(B@B),name='B') #psi stuff

# Main solving loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info(f'Completed iteration: {solver.iteration}')
            logger.info(f'Simulation Time: {solver.sim_time:.4f}')
            logger.info(f'dt: {timestep:.5f}')
            logger.info(f"Max B = {flow.max('B'):.5f}, Max v = {flow.max('v'):.5f}\n")
            logger.info('----------------------------------------\n')

            if flow.max('B') > 3 or np.any(np.isnan(flow.max('B'))): # Stop condition for instabilities
                if CW.rank == 0:
                        logger.error('Numerical instability detected, ending simulation')
                logger.info(f"Max B = {flow.max('B')}")
                logger.info(f"Max v = {flow.max('v')}")
                logger.info(f"NANs in B: {np.any(np.isnan(flow.max('B')))}")
                logger.info(f"NANs in v: {np.any(np.isnan(flow.max('v')))}\n")
                logger.info('---------------------------------------------\n')
                exit()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
