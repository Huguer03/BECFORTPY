import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from cooling import Grid, Simulation, ThomasFermi
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 21,   
    'ytick.labelsize': 21,    
    'legend.fontsize': 23,    
    'axes.grid': False        
})

def graf_densidad(L, density, t):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
    zoom_region = [-L[0]/6, L[0]/6, -L[0]/6, L[0]/6]

    im = ax.imshow(density, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax.axis(zoom_region)
    ax.set_xticks([])   
    ax.set_yticks([])

    fig.colorbar(im, ax=ax)

    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/fase_mascara_cool/{str(t)}_densidad.png', dpi=300)
    plt.close()

def graf_phase(L, phase, t):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
    zoom_region = [-L[0]/6, L[0]/6, -L[0]/6, L[0]/6]

    im = ax.imshow(phase, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='Greys')
    ax.axis(zoom_region)
    ax.set_xticks([])   
    ax.set_yticks([])

    fig.colorbar(im, ax=ax)

    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/fase_mascara_cool/{str(t)}_phase.png', dpi=300)
    plt.close()

def test():
    beta = 500.0    
    gamma = (10.0, 10.0)
    sigma = (1.0,1.0)
    Omega = 0.2    
    tf = ThomasFermi(gamma, beta)
    N = (2**8, 2**8)
    L = (10*tf.rtf, 10*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L, L[0]/N[0])
    n_vortex = 1
    tol = 1e-13
    vortex_charge = [1]
    positions = [
        (0.0, 0.0)
    ]

    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     sigma         = sigma,
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = n_vortex, 
                     vortex_charge = vortex_charge, 
                     positions     = positions
                     )
    

    converge = False
    i = 0
    while converge == False:
        converge = sim.cooling(1e-4, tol=tol, max_iter=1000)
        graf_densidad(L, sim.wf.density(), i)
        graf_phase(L, sim.wf.phase(), i)
        i +=1

if __name__ == "__main__":
    test()