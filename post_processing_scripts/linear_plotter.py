#for plotting linear eigenmodes
import h5py
import numpy as np
from matplotlib import pyplot as plt
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD

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

#Now the actual file
run_num = '360'
Nx = 256
Lx = np.pi
f = h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx}/eigenmodes/{run_num}_{Nx}_right.h5', 'r') #open up the data to read
# f = h5py.File(f'/home/d3test/main/linear/{Nx+1}/eigenmodes/{run_num}_{Nx+1}_right.h5', 'r') #open up the data to read
psi = f['tasks/psi']
# phi = f['tasks/phi']
gam = f['scales/growth_rate']
frequency = f['scales/frequency']
Ev = f['scales/complex_EV']


xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.complex128)
xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=1)
x = dist.local_grid(xbasis,scale=1)

grid_psi = np.zeros((Nx*2,Nx),dtype=np.complex128)
print(np.shape(psi))
print(np.shape(grid_psi))

delete=[]
for eigen_ind, value in enumerate(psi[:,0]):
    grid_psi[eigen_ind,:] = convert_data_1d(psi[eigen_ind,:],in_format='c',out_format='g')


print(grid_psi[0,:])

grid_psi = np.delete(grid_psi, delete, axis=0)
gam = np.delete(gam,delete,axis=0)

print(f' max max {np.max(gam)}')
# psi_g = convert_data_1d(psi[np.argmax(gam),:],Nx,Lx)
plt.plot(x,grid_psi[np.argmin(gam)], label = 'real')
# plt.plot(x,psi_g.imag, label = 'imag')
plt.xlabel('$x$')
plt.ylabel('$\psi \ Amplitude$')
# plt.legend()
plt.title(f'Psi vs x from Eigenmode Run {run_num}  Eigenmodes')
# plt.savefig(f'/home/d3test/main/linear/{Nx+1}/mode/psi/{run_num}_most_unstable.png', format='png')
# plt.savefig(f'/home/d3test/main/linear/{Nx+1}/mode/psi/{run_num}_psi.png', format='png')
plt.savefig(f'/home/d3test/main/linear/{Nx}/mode/psi/{run_num}_psi.png', format='png')
plt.show()
plt.close()


# plt.plot(x,phi[np.argmax(gam)], label = 'real')
# # plt.plot(x,psi_g.imag, label = 'imag')
# plt.xlabel('x')
# plt.ylabel('$\phi \ Amplitude$')
# # plt.legend()
# plt.title(f'Most Unstable Phi vs x from Eigenmode Run {run_num}  Eigenmodes')
# plt.savefig(f'/home/d3test/main/linear/{Nx+1}/mode/phi/{run_num}_most_unstable.png', format='png')
# plt.show()
# plt.close()

# psi_g = convert_data_1d(psi[np.argmin(gam),:],Nx,Lx)
# plt.plot(x,psi_g.real, label = 'real')
# plt.plot(x,psi_g.imag, label = 'imag')
# plt.xlabel('x')
# plt.ylabel('$\psi \ Amplitude$')
# plt.legend()
# plt.title(f'Most Stable Psi vs x from Eigenmode Run {run_num}')
# plt.savefig(f'/data_storage/Plasma_Research/data_stolnicki/Ottaviani_Porcelli_93/linear/{Nx}/mode/psi/{run_num}_most_stable.png', format='png')
# plt.show()
# plt.close()

# phi_g = convert_data_1d(phi[np.argmax(gam),:],Nx,Lx)
# plt.plot(x,phi_g.real, label = 'real')
# plt.plot(x,phi_g.imag, label = 'imag')
# plt.xlabel('x')
# plt.ylabel('$\psi \ Amplitude$')
# plt.legend()
# plt.title(f'Most Unstable Phi vs x from Eigenmode Run {run_num}')
# plt.savefig(f'/data_storage/Plasma_Research/data_stolnicki/Ottaviani_Porcelli_93/linear/{Nx}/mode/phi/{run_num}_most_unstable.png', format='png')
# plt.show()
# plt.close()

# phi_g = convert_data_1d(psi[np.argmin(gam),:],Nx,Lx)
# plt.plot(x,phi_g.real, label = 'real')
# plt.plot(x,phi_g.imag, label = 'imag')
# plt.xlabel('x')
# plt.ylabel('$\psi \ Amplitude$')
# plt.legend()
# plt.title(f'Most Stable Phi vs x from Eigenmode Run {run_num}')
# plt.savefig(f'/data_storage/Plasma_Research/data_stolnicki/Ottaviani_Porcelli_93/linear/{Nx}/mode/phi/{run_num}_most_stable.png', format='png')
# plt.show()
# plt.close()
