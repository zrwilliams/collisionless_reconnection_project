import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from mpi4py import MPI
CW = MPI.COMM_WORLD



# Just for reading in data easy
Nx = '1024'
run_num = '328'


# Read in each variable from both L and R eigenmode file
with h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx}/eigenmodes/{run_num}_{Nx}_right.h5',mode='r') as file:
    psi_r = np.array(file['tasks/psi'])
    phi_r = np.array(file['tasks/phi'])
    U_r = np.array(file['tasks/U'])

with h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx}/eigenmodes/{run_num}_{Nx}_left.h5',mode='r') as file:
    psi_l = np.array(file['tasks/psi'])
    phi_l = np.array(file['tasks/phi'])
    U_l = np.array(file['tasks/U'])


print(f"phi Shape: {np.shape(phi_r)}, psi Shape: {np.shape(psi_r)}")

kronecker = np.zeros((len(psi_r[:,0]),len(psi_r[:,0])))
for i in range(len(psi_r[:,0])):
    if i%50 == 0:
        print(f"{i}/{len(psi_r[:,0])}")
    for j in range(len(psi_r[:,0])):
        kronecker[i,j] = np.abs(np.transpose(phi_l[i]).conj()@np.transpose(phi_r[j]) + np.transpose(psi_l[i]).conj()@np.transpose(psi_r[j]) + np.transpose(U_l[i]).conj()@np.transpose(U_r[j]))

# print(kronecker[256,:])
# kronecker = np.tensordot(np.conj(data_l), data_r, axes=([0,2],[0,2]))
# print(f"k-shape {np.shape(kronecker)}")
#kronecker = np.tensordot(data_l, data_r, axes=([0,2],[0,2])) # tensordot without conjugate lefts - believe dedalus does this under the hood 





# Plot
plt.matshow(np.abs(kronecker),norm=matplotlib.colors.LogNorm())
# plt.matshow(np.abs(kronecker))
plt.colorbar()
plt.title(f'Orthogonality Check Linear {run_num} ')
plt.savefig(f'/home/d3test/main/linear/orthog/{run_num}_orthogonality.png',format='png',dpi=150,bbox_inches='tight')     
plt.close()

