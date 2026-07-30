import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from mpi4py import MPI
import pyparsing
CW = MPI.COMM_WORLD
import h5py

# LIN = input("\nPlotting linear? (True/False): ")
# if str.lower(LIN)=='true':
#     LIN = True                                                                      
# else:
#     LIN = False
# NONLIN = input("Plotting nonlinear? (True/False): ")
# if str.lower(NONLIN)=='true':
#     NONLIN = True                                                                      
# else:
#     NONLIN = False

LIN = True
NONLIN = False
tind = 117

#linear first
def convert_data_1d(data, Nx, Lx, in_format='g', out_format='c'):
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.complex128) 
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

if LIN == True:
    #set up grid
    lin_run = '220'
    beta_run = '223'
    Nx = 1023
    Lx = np.pi

    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.complex128) 
    xbasis = d3.ComplexFourier(xcoord, Nx, bounds=(-Lx, Lx),dealias=3/2)
    x = dist.local_grid(xbasis)


    #Right eigenmode file                                                                               
    with h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/linear/{Nx+1}/eigenmodes/{lin_run}_{Nx+1}_right.h5', 'r') as f:                                                  
        psi = np.array(f['tasks/psi'])
        Ev = np.array(f['scales/complex_EV'])

    #beta file
    with h5py.File(f'/media/williams/plasma_data/collisionless_tearing_project/2024/betas/beta_data/{beta_run}_{Nx+1}_betas.h5', mode='r') as file:
        betasL = np.array(file['betas0'])
        beta_tind = tind # What tind to grab betas from
        betas = betasL[beta_tind,:]
    
    print(f"Intial Beta Shape: {np.shape(betas)}, Intial Psi Shape: {np.shape(psi)}, Intial Ev Shape: {np.shape(Ev)}")

    
    delete_list = []
    for index,value in enumerate(np.abs(betasL[-1,:])):
        if value < 0.001:
            delete_list.append(index)
    betas = np.delete(betas, delete_list)
    psi = np.delete(psi, delete_list, axis=0)

    print(f"Final Beta Shape: {np.shape(betas)} Final Psi Shape: {np.shape(psi)}")

    psi_sum = np.zeros_like(psi[0,:], dtype=complex)
    #iterate over eigenmodes
    for index, value in enumerate(psi):
        psi_current = np.multiply(psi[index,:],betas[index])
        psi_sum = np.add(psi_sum,psi_current)

    # psi_sum = np.multiply(psi[np.argmax(betas),:],betas[np.argmax(betas)])
    #iterate over select eigenmodes
    # print(indicies)
    # for index, value in enumerate(indicies):
    #     psi_current = np.multiply(psi[value,:],betas[value])
    #     psi_sum = np.add(psi_sum,psi_current)
    
    
    print(f"Psi sum shape: {np.shape(psi_sum)}")
    # print(psi_sum)
    psi_sum = convert_data_1d(psi_sum, Nx, Lx, in_format='c', out_format='g')
    print(f"Psi sum shape: {np.shape(psi_sum)}")
    print(psi_sum)

    plt.style.use('classic')
    plt.plot(x, psi_sum)
    # plt.xlim(-0.1,0.1)
    # plt.ylim(-0.1,0.1)
    # plt.title('Summation of Psi Eigenmode * Beta at Time -1')
    plt.savefig(f'/home/d3test/main/betas/nonlin_lin/{beta_run}/linear_t{tind}.png', format='png')
    plt.show() 
    plt.close()





#Doing nonlinear now 
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

def mode_v_x(field, axis, ky_index, Ny, Nx, Ly, Lx, tind):
    field_c = convert_data_2d(np.array(field[tind,:,:]),Ny, Nx, Ly, Lx)
    field_1dg = convert_data_1d(field_c[ky_index,:], Nx, Lx, in_format='c', out_format='g')
    # print(np.shape((axis)))
    plt.plot(axis, field_1dg)
    plt.savefig(f'/home/d3test/main/betas/nonlin_lin/{beta_run}/nonlin_t{tind}.png',format='png')
    plt.close()
    # plt.semilogy(axis, field_1dg)

if NONLIN == True:
    Ny = 32
    Nx = 511
    Ly = 2*np.pi
    Lx = np.pi
    kyi = 3
    nonlin_run = '198'

    plt.style.use('classic') 

    coords = d3.CartesianCoordinates('y','x')
    dist2 = d3.Distributor(coords, dtype=np.complex128)
    xbasis2 = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx, Lx),dealias=3/2)
    ybasis2= d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly, Ly),dealias=3/2)


    x = dist.local_grid(xbasis)
    with h5py.File(f'/home/d3test/main/nonlinear/{Nx+1}/snapshots/{nonlin_run}_nonlinear_sim/{nonlin_run}_nonlinear_sim_s1.h5',mode='r') as f:
        psin = np.array(f['tasks/psi'])

    mode_v_x(field=psin,axis=x,ky_index=kyi, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, tind=tind)