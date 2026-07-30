import h5py
import numpy as np
from matplotlib import pyplot as plt
import dedalus.public as d3
import scipy
import scipy.optimize


#This function takes a 2d field in grid space ('g') and converts it to coefficient space ('c'). It needs to be called for every timestep
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
def convert_data_1d(data, Nx, Lx, in_format='c', out_format='g'):
    data_field = dist.Field(name='data',bases=xbasis)
    if out_format=='g':
        dtypeout=np.float64
    else:
        dtypeout=np.complex128
    data_field[in_format] = data
    data_out = np.zeros(np.shape(data_field[out_format]), dtype=dtypeout)
    data_out = data_field[out_format]
    return data_out


# Specify simulation box parameters
Ny = 64
Nx = 1023
Ly = 2*np.pi 
Lx = np.pi 
nonlin_run = '367'
lin_run = '220'
beta_run = '368'
ky_index = 1

# index 99 is stable mode for run 223 98 is unstable
# index 598 is stable mode for run 239 599 is unstable
# index 13 is stable mode for run 239 12 is unstable
# or just pull min/max from gam its easier

# O-point ind = 512, 511 ; X-point ind = 0,-1

Nonlinear_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/nonlinear/{Nx+1}/snapshots/{nonlin_run}_nonlinear_sim/{nonlin_run}_nonlinear_sim_s1.h5'
Right_Eigen_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx+1}/eigenmodes/{lin_run}_{Nx+1}_right.h5'
Beta_Save_Path = f'/media/williams/plasma_data/collisionless_tearing_project/2024/betas/beta_data/{beta_run}_{Nx+1}_betas.h5'


coords = d3.CartesianCoordinates('y','x')
dist2 = d3.Distributor(coords, dtype=np.complex128)
xbasis2 = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx, Lx),dealias=3/2)
ybasis2= d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly, Ly),dealias=3/2)

xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.complex128) 
xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=3/2)
x = dist.local_grid(xbasis)
#extract needed information from the h5 file
with h5py.File(Nonlinear_Path, 'r') as f:
    psiNL = np.array(f['tasks/psi'] )
    t = np.array(f['scales/sim_time'])
    t = t[0:t.size-4]
with h5py.File(Beta_Save_Path, 'r') as f:
    betas = np.array(f['betas0'])
with h5py.File(Right_Eigen_Path) as f:
    psiLI = np.array(f['tasks/psi'])
    growth = np.array((f['scales/growth_rate']))
    stable_ind = np.argmin(growth)

plt.style.use('classic') #personal preference 

t_amount = len(t)
field_t = np.zeros(t_amount) 
growth_func = np.zeros(t_amount)
psi_noS = np.zeros(t_amount)

x_psi = np.zeros(t_amount)
x_psi_noS = np.zeros(t_amount)
o_psi = np.zeros(t_amount)
o_psi_noS = np.zeros(t_amount)


for i in range(t_amount): #Performs the conversion at each time step

    ### Calculating with average over x ###
    field_c = convert_data_2d(psiNL[i,:,:],Ny, Nx, Ly, Lx) #converts original field (psi or phi) to 'c' space
    field_1dg = convert_data_1d(field_c[ky_index,:], Nx, Lx) #Chooses index of user-selected ky, converts that 1d array back to 'g' space
    int_field = np.sqrt((np.trapz(np.abs(field_1dg*field_1dg),x)).real) #this line and the next are just for averaging in the chebyshev direction; take an integral...
    field_t[i] = int_field #abs(avg_field) #taking the absolute value just so the numbers are positive for plotting. Also, when converting back to 'g', the quantity becomes real again, which is why a couple lines up I have .real (the complex part should be 10^-17 or less)
   
    growth_func[i] = 0.0005*np.exp(0.04614947491698067*t[i]) #exponential function

    pre_stable_psi = np.multiply(betas[i,stable_ind],psiLI[stable_ind,:]) 
    stable_psi = convert_data_1d(pre_stable_psi, Nx, Lx)

    psi_noS[i] = np.sqrt((np.trapz(np.abs((field_1dg-stable_psi)*(field_1dg-stable_psi)),x)).real)


    ### Calculating at x & o points ###

    # x_psi[i] = np.abs(field_1dg[0])
    # o_psi[i] = np.abs(field_1dg[511])

    # x_psi_noS[i] = np.abs(field_1dg[0] - stable_psi[0])
    # o_psi_noS[i] = np.abs(field_1dg[511] - stable_psi[511])



# #### Integral Plotting ####
plt.semilogy(t, field_t,label='$\psi_{NL}$') # NL Psi plot
plt.semilogy(t, growth_func,linestyle='dotted', label='$e^{\gamma t}$')
plt.semilogy(t, psi_noS, linestyle='dashed', color='red', label='$\psi_{mod}$')
# plt.ylim(1e-4,1)
# plt.xlim(20,t_amount)

#### X-O point Plotting ####
# plt.semilogy(t, x_psi,label='full $\psi_x $')
# plt.semilogy(t, x_psi_noS,label='no stable $\psi_x $', linestyle='dashed')
# plt.semilogy(t, o_psi,label='full $\psi_o $')
# plt.semilogy(t, o_psi_noS,label='no stable $\psi_o $', linestyle='dashed')
# plt.ylim(1e-4,1e-1)
# plt.xlim(20,t_amount)

# plt.title(f'Root mean square case - run {beta_run}')
plt.ylabel(r'$\mathrm{Magnitude}$',fontsize=25)
plt.xlabel(r'$t/\tau_A$',fontsize=25)
plt.legend(loc='lower right',fontsize=24)
plt.savefig(f'/home/d3test/main/betas/deviations/{beta_run}_rms_deviations.png', format='png', bbox_inches='tight')

