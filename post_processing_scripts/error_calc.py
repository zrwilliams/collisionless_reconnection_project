import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from mpi4py import MPI
import pyparsing
CW = MPI.COMM_WORLD
import h5py

def convert_data_2d(data, Ny, Nx, Ly, Lx, informat='g', outformat='c'):
    data_field = dist2.Field(name='data',bases=(xbasis2,ybasis2))
    if outformat=='g':
        dtypeout=np.float64
    else:
        dtypeout=np.complex128
    data_field[informat] = data
    data_out = np.zeros(np.shape(data_field[outformat]), dtype=dtypeout)
    data_out = data_field[outformat]
    return data_out

#This function takes a 1d field in coefficient space ('c') and converts it back to grid space ('g'). Similar to the previous function, just in 1D
def convert_data_1d(data, Nx, Lx, informat='c', outformat='g'):
    data_field = dist.Field(name='data',bases=xbasis)
    if outformat=='g':
        dtypeout=np.float64
    else:
        dtypeout=np.complex128
    data_field[informat] = data
    data_out = np.zeros(np.shape(data_field[outformat]), dtype=dtypeout)
    data_out = data_field[outformat]
    return data_out


def calculate_psi_sum(mode, tind, Nx, Lx):
    # with h5py.File(Right_Eigen_Path, 'r') as f:                                                  
    #     psi = np.array(f['tasks/psi'])
    #     Ev = np.array(f['scales/complex_EV'])

    with h5py.File(Betas_Path, mode='r') as file:
        betasL = np.array(file['betas0'])
        betas = betasL[tind,:]

    # print(f"Initial Beta Shape: {np.shape(betas)} ")
    # for filtering large/small beta values
    if filter:
        delete_list = []
        for index,value in enumerate(np.abs(betasL[0,:])):
            if value < threshold:
                delete_list.append(index)
            # elif index == np.argmax(gam):
            #     print(f"deleting beta w gam={np.min(gam)} at index ={np.argmin(gam)}")
            #     delete_list.append(index)
        betas = np.delete(betas, delete_list)
        mode = np.delete(mode, delete_list, axis=0)
    
    # print(f"Final Beta Shape: {np.shape(betas)}")

    mode_sum = np.zeros_like(mode[0,:], dtype=complex)
    for index, value in enumerate(mode):
        mode_current = np.multiply(mode[index,:],betas[index])
        mode_sum = np.add(mode_sum,mode_current)
    # print(f"Psi sum shape: {np.shape(psi_sum)}")
    mode_sum = convert_data_1d(mode_sum, Nx, Lx, informat='c', outformat='g')
    # print(mode_sum)
    return mode_sum




def calculate_psi_NL(mode,tind, kyi, Nx, Lx, Ny, Ly):
    f = h5py.File(Nonlinear_Path, 'r') #open up the data to read
    psiNL = f[f'tasks/{mode}']
    # print(f"Start: {np.shape(psiNL)}")
    psiNL = convert_data_2d(np.array(psiNL[tind,:,:]), Ny, Nx, Ly, Lx, informat='g', outformat='c')
    # print(f"Second: {np.shape(psiNL)}")
    psiNL = convert_data_1d(psiNL[kyi,:], Nx, Lx,informat='c',outformat='g')
    # print(f"Final {np.shape(psiNL)}")
    return psiNL



Nx = 1023
Ny = 64
Lx = np.pi
Ly = 2*np.pi
kyi = 1
threshold = 1e-4 # for cutting off modes 
nonlin_run = '222'
lin_run = '220'
beta_run_num = '223'
filter = True

Nonlinear_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/nonlinear/{Nx+1}/snapshots/{nonlin_run}_nonlinear_sim/{nonlin_run}_nonlinear_sim_s1.h5'
Right_Eigen_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx+1}/eigenmodes/{lin_run}_{Nx+1}_right.h5'
Betas_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/betas/beta_data/{beta_run_num}_{Nx+1}_betas.h5'

if filter:
    Save_Path = f"/home/d3test/main/betas/error/{beta_run_num}_pair_error.png"
else:
    Save_Path = f"/home/d3test/main/betas/error/{beta_run_num}_full_error.png"


    
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.complex128) 
xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=3/2)
x = dist.local_grid(xbasis)


coords = d3.CartesianCoordinates('y','x')
dist2 = d3.Distributor(coords, dtype=np.complex128)
xbasis2 = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx, Lx),dealias=3/2)
ybasis2= d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly, Ly),dealias=3/2)

with h5py.File(Betas_Path, 'r') as f:
    sim_time = np.array(f['betas0'])[:,0].size
plt.style.use('classic')

for index in [0,1]: # this if want both psi and phi [0,1]:
    Error = []
    if index == 0:
        with h5py.File(Right_Eigen_Path, 'r') as f:                                                  
            mode = np.array(f['tasks/psi'])
            gam = np.array(f['scales/growth_rate'])
            task = 'psi'
    else:
        with h5py.File(Right_Eigen_Path, 'r') as f:                                                  
            mode = np.array(f['tasks/phi'])
            task = 'phi'
    for tind in range(sim_time):
        if tind % 20 == 0 :
            print(f"Solving for time index: {tind}/{sim_time}")
        psiNL = calculate_psi_NL(task,tind, kyi, Nx, Lx, Ny, Ly)
        psi_sum = calculate_psi_sum(mode,tind, Nx, Lx)
        difference = np.multiply((psiNL - psi_sum),(psiNL - psi_sum))
        integral_top = (np.trapz((difference), x=x)) / Lx #removed sqrt
        integral_bot = (np.trapz(np.multiply(psiNL,psiNL),x=x))/Lx #removed sqrt
        error_current = np.divide(integral_top, integral_bot)
        Error.append(error_current.real*100)
        # print(integral_top)
        # print(error_current)
        
    if index == 0:
        plt.semilogy(Error,label='$\mathrm{Error \: in \: \Psi}$')
    else:
        plt.semilogy(Error,label='$\mathrm{Error \: in \: \Phi}$')

plt.style.use('classic')
# plt.semilogy(Error)
plt.legend(loc=0,fontsize=20)#(loc='center right')
plt.xlabel(r'$t/\tau_A$',fontsize=25)
plt.ylabel("$\mathrm{Err}(t)$",fontsize=25)
#plt.ylim(-10,600)
# plt.xlim(0,150)
plt.savefig(Save_Path, format='png',bbox_inches='tight')
