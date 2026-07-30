from ast import Index
from pydoc import ModuleScanner
import h5py
import numpy as np
from matplotlib import pyplot as plt

import dedalus.public as d3

#This function takes a 2d field in grid space ('g') and converts it to coefficient space ('c'). It needs to be called for every timestep
def convert_data_2d(data, informat='g', outformat='c'):
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
def convert_data_1d(data, in_format='c', out_format='g'):
    data_field = dist.Field(name='data',bases=xbasis)
    if out_format=='g':
        dtypeout=np.float64
    else:
        dtypeout=np.complex128
    data_field[in_format] = data
    data_out = np.zeros(np.shape(data_field[out_format]), dtype=dtypeout)
    data_out = data_field[out_format]
    return data_out

#This function takes the desired field input, the chebyshev axis, which ky index you want to evaluate, and simulation domain details as input
def mode_v_time(field, axis, ky_index, t,modename):
    # print(f' Shape is {np.shape(np.array(field))}')
    t_amount = len(t)
    field_t = np.zeros(t_amount) 
    for i in range(t_amount): #Performs the conversion at each time step
        field_c = convert_data_2d(np.array(field[i,:,:])) #converts original field (psi or phi) to 'c' space
        # print(np.shape(field_c))
        field_1dg = convert_data_1d(field_c[ky_index,:]) #Chooses index of user-selected ky, converts that 1d array back to 'g' space
        int_field = (np.trapz(np.abs(field_1dg),axis)).real #this line and the next are just for averaging in the chebyshev direction; take an integral...
        field_t[i] = int_field/Lx #abs(avg_field) #taking the absolute value just so the numbers are positive for plotting. Also, when converting back to 'g', the quantity becomes real again, which is why a couple lines up I have .real (the complex part should be 10^-17 or less)
        # print(field_t[i])
    # plt.plot(t,field_t,label='$k_y = {0:.2f}$'.format(ky_index*(2*np.pi/(2*Ly)))) #after looping through all the time units for this specific ky, plot it. Whitespace is important!
    if modename =='psi':
        # plt.plot(t, field_t,label='$\Psi k_y = {0:.2f}$'.format(ky_index*(2*np.pi/(2*Ly))))
        plt.semilogy(t, field_t,label='$\Psi k_y = {0:.2f}$'.format(ky_index*(2*np.pi/(2*Ly))))
    else:
        plt.semilogy(t, field_t,linestyle='dashed',label='$\Phi k_y = {0:.2f}$'.format(ky_index*(2*np.pi/(2*Ly))))




# Specify simulation box parameters
Ny = 64
Nx = 1023
Ly = 2*np.pi 
Lx = np.pi 
run_num = '371'

f = h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/nonlinear/{Nx+1}/snapshots/{run_num}_nonlinear_sim/{run_num}_nonlinear_sim_s1.h5', 'r') #open up the data to read

coords = d3.CartesianCoordinates('y','x')
dist2 = d3.Distributor(coords, dtype=np.complex128)
xbasis2 = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx, Lx),dealias=3/2)
ybasis2= d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly, Ly),dealias=3/2)

xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.complex128) 
xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=3/2)
x = dist.local_grid(xbasis)
#extract needed information from the h5 file
psi = f['tasks/psi'] 
phi = f['tasks/phi']
t = np.array(f['scales/sim_time'])
# print(f"Sim time shape {np.shape(t)}")


plt.style.use('classic') #personal preference


modes = [psi, phi]
mode_names = ['psi','phi']

for index, mode  in enumerate(modes):
    # Plots psi v time for chosen ky values
    print(f"Reading in {mode_names[index]}")
    for ky_ind in range(1,4): #just plotting six modes. ky = 0 contains the equilibrium and is WAY larger than the rest, so I don't plot it 
        # print(f'Plotting time index {n}')
        mode_v_time(mode, x, ky_ind, t, modename=mode_names[index]) 
    # plot details...
    plt.legend(loc='lower right')
    plt.title(f"Time Evolution of Nonlinear Dynamic Variables from Run {run_num}")
    plt.xlabel('$\omega_A t$',fontsize=23)
    plt.ylabel('$Mode Amplitude$',fontsize=23)
plt.savefig(f'/home/d3test/main/nonlinear/{Nx+1}/mode_v_time/{run_num}_semilog_nonlinear_modes_v_t.png',format='png')
plt.show()
plt.close()

# for index, mode  in enumerate(modes):
#     mode_v_x(field=mode,mode_name = mode_names[index], axis=x,ky_index=1, tind=-1)

