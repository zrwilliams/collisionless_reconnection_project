import numpy as np
import h5py
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import sys
import time
import logging
logger = logging.getLogger(__name__)

###########################################################################################################
#### CHECK KY INDEX. Min ky from nonlinear is pi/Ly. Will NOT work if there is mismatch in wavenumber #####
###########################################################################################################

# Parameters
kyi = 1 # y wavenumber index 
Ny = 64
Nx = 1023
Ly = 2*np.pi
Lx = np.pi
# Input nonlinaer and linear simulation run numbers for reading in data
nonlin_run = '371'
lin_run = '220'
beta_run = '372'

Nonlinear_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/nonlinear/{Nx+1}/snapshots/{nonlin_run}_nonlinear_sim/{nonlin_run}_nonlinear_sim_s1.h5'
Right_Eigen_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx+1}/eigenmodes/{lin_run}_{Nx+1}_right.h5'
Left_Eigen_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx+1}/eigenmodes/{lin_run}_{Nx+1}_left.h5'
Beta_Save_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/betas/beta_data/{beta_run}_{Nx+1}_betas.h5'

try:
    Verbose = str(sys.argv[1])
except IndexError:
    Verbose = 'F'

# For converting the 1d data from grid to coefficient space or vice versa 
def fourier_convert_data_1d(data, Nx, Lx, in_format='g', out_format='c'):
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.complex128,comm=MPI.COMM_SELF) #make sure to have mpi.COMM_SELF for accurate parallelization
    xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=3/2)
    data_field = dist.Field(name='data',bases=xbasis)
    if out_format=='g':
        dtypeout=np.float64
    else:
        dtypeout=np.complex128
    data_field[in_format] = data
    data_out = np.zeros(np.shape(data_field[out_format]), dtype=dtypeout)
    data_out = data_field[out_format]
    return data_out

# For converting the 2d data from grid to coefficient space or vice versa 
def convert_data_once(data, Nx, Ny, Lx, Ly, informat='g', outformat='c'):
    coords = d3.CartesianCoordinates('y','x')
    dist = d3.Distributor(coords, dtype=np.complex128,comm=MPI.COMM_SELF)
    xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx, Lx),dealias=3/2)
    ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly, Ly),dealias=3/2)

    data_field = dist.Field(name='data',bases=(ybasis,xbasis))
    if outformat=='g':
        dtypeout=np.float64
    else:
        dtypeout=np.complex128
    data_field[informat] = data
    data_out = np.zeros(np.shape(data_field[outformat]), dtype=dtypeout)
    data_out = data_field[outformat]
    return data_out


# For reading in data from nonlinear simulation at a given time index, converts data to coefficient space 
def data_NL_from_run(Nx, Ny, Lx, Ly, tind):
    data_NL = []
    NL_tasks = ['phi','psi'] #Note: all derivatives must have x as last char in string. Change line 94 and 95 if deriving with respect to different variable
    with h5py.File(Nonlinear_Path, mode='r') as file:
        for task in NL_tasks:
            data_NL.append(convert_data_once(np.array(file['tasks/'+task][tind]), Nx, Ny, Lx, Ly, informat='g', outformat='c'))
    
    return np.array(data_NL) # Outputs an array with dimensions (task, y grid, x grid)

# For reading in linear data, automatically read in coefficient space
def quick_linear_data():
    with h5py.File(Right_Eigen_Path,mode='r') as file:
        psi_r = np.array(file['tasks/psi'])
        phi_r = np.array(file['tasks/phi'])

    with h5py.File(Left_Eigen_Path,mode='r') as file:
        psi_l = np.array(file['tasks/psi'])
        phi_l = np.array(file['tasks/phi'])

    return psi_r, phi_r, psi_l, phi_l

# Main function for calculating betas, called for every time index (tind) in nonlinear simulation
def get_betas(kyi, tind, Nx, Ny, Lx, Ly, n):
    if CW.rank == 0 and tind % 10 == 0: # First thread outputs calculation status
        logger.info(f"Thread 0: Solving for time index {tind}/{int(len(time_global)/CW.size)}")
    dataNL_raw = data_NL_from_run(Nx,Ny, Lx, Ly,tind) # get nonlinaer data  
    psi_r, phi_r, psi_l, phi_l = quick_linear_data() # get linear data
    betas = np.zeros(n, dtype=np.complex128) 
    norm = np.zeros(n, dtype=np.complex128)
    

    for i in range(len(psi_r[:,0])): # For each eigenmode, compute the inner products of the left eigenmodes with the nonlinear data at the chosen y wavenumber
        betas[i] =  np.transpose(phi_l[i,:]).conj()@np.transpose(dataNL_raw[0,kyi,:]) \
                    + np.transpose(psi_l[i,:]).conj()@np.transpose(dataNL_raw[1,kyi,:])

        # Calculate norm as the product of the left eigenmodes with the rights
        norm[i] =  np.transpose(phi_l[i]).conj()@np.transpose(phi_r[i]) \
                   + np.transpose(psi_l[i]).conj()@np.transpose(psi_r[i])
        
    betas = np.divide(betas,norm) # Divide raw beta data by norm 
    
    return betas




if True:
    with h5py.File(Nonlinear_Path, mode='r') as file:
        sim_time = np.array(file['scales/sim_time'])
        max_tind = len(sim_time) # Choosing max time index such that betas are calculated for every time index in nonlinear simulation. For efficient parallelization, powers of 2 are best

    with h5py.File(Right_Eigen_Path, mode='r') as file:
        eigenmode_num = (np.shape(file['tasks/phi'])[0]) # Get number of eigenmodes from linear data 

    time_global = np.arange(max_tind)

    if CW.rank == 0:
        print(f"Solving for ky of {kyi*(np.pi/(Ly))}")
        print(f"Global Index Length of {max_tind}")


    time_global = np.arange(max_tind)

    # Give each thread a subsection of time indicies to calculate betas for 
    start = int((CW.rank*len(time_global)/CW.size))
    stop = int(((CW.rank+1)*len(time_global)/CW.size))

    #Splits time global into time local where each thread contains the set of tind they are calculating for
    time_local = np.arange(start, stop)
    t1 = time.time()
    beta_local = [(np.array(get_betas(kyi, tind ,Nx, Ny, Lx, Ly,eigenmode_num))) for tind in time_local]
    t2 = time.time()
    logger.info(f'Finished solving betas, elapsed solve time: {t2-t1}')

    beta_global = np.array(CW.gather(beta_local)) # Gather beta data from each thread into one large array 

    if CW.rank == 0: # First thread saves beta data in h5py file 
        two_d_betas = beta_global.reshape(-1,(eigenmode_num))

        with h5py.File(Beta_Save_Path, mode='w') as file:
            dset1 = file.create_dataset('sim_time', data=sim_time)
            dset2 = file.create_dataset('betas0', data=two_d_betas)
            # print(np.shape(two_d_betas))
        print('Finished solving for betas')







