import h5py
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD



Nx = 1023
beta_run_num = '372'
lin_run_num = '221'
tind = -1
filter = False

with h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/betas/beta_data/{beta_run_num}_{Nx+1}_betas.h5', mode='r') as file:
    betasL = np.array(file['betas0'])
    betas = betasL[tind,:]

        
with h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx+1}/eigenmodes/{lin_run_num}_{Nx+1}_right.h5', mode='r') as file:
    Growth = np.array((file['scales/growth_rate']))
    Frequency = np.array((file['scales/frequency']))
    Ev = np.array((file['scales/complex_EV']))

print(np.shape(betas))
s_ind = np.argmin(Growth)
print(np.abs(betas[s_ind]))

print(s_ind)
# number = 0.04217145514768533
# asd = np.abs(betas)
# print(np.where(asd == number)[0])

if filter:
    list = []
    for index,value in enumerate(np.abs(betasL[-1,:])):
        if value > 1e101:
            list.append(index)
        # elif value > 0.1:
        #     list.append(index)
    betas = np.delete(betas, list)
    Growth = np.delete(Growth, list)
    Frequency = np.delete(Frequency, list)
    Ev = np.delete(Ev, list)





length = 0
for i  in range(len(Growth)):
    if np.isfinite(Frequency[i]):
        length += 1



three_d_array = np.zeros((3, length),dtype = np.complex128)
j = 0

for i in range(length):
    if np.isfinite(Frequency[i]):
        three_d_array[0][j] = Growth[i]
        three_d_array[1][j] = Frequency[i]
        three_d_array[2][j] = np.abs(betas[i]) #*np.sqrt(energies[i])
        j +=1
    else:
        # print('F: ', Frequency[i], ' G: ', Growth[i], ' B: ',betas[i])
        Frequency_f = np.delete(Frequency, i)
        Growth_f = np.delete(Growth, i)
        betas_f = np.delete(betas, i)



norm=matplotlib.colors.LogNorm()
plt.style.use('classic')
plt.scatter(three_d_array[0],three_d_array[1],s=100,c=three_d_array[2],cmap='cool',edgecolors='none',norm=norm)
plt.colorbar().set_label(label='$\mathrm{Eigenmode \ Amplitudes}$',fontsize=20)
plt.xlabel(r'$\gamma\tau_A$',fontsize=25)
plt.ylabel(r'$\omega_r\tau_A$',fontsize=25)
# plt.title(f'Linear Eigenspace Including Betas from {beta_run_num} at Time Index {tind}')
plt.tight_layout()
plt.grid(True,which='both')
plt.axhline(linewidth=1,y=0,color='k')
plt.axvline(linewidth=1,x=0,color='k')
# plt.xlim(-0.75,0.1)
# plt.ylim(-0.45,0.45)
plt.savefig(f"/home/d3test/main/betas/eigenspace_beta/{beta_run_num}_eigenspace.png",format='png',bbox_inches='tight')
plt.show()
plt.close()
