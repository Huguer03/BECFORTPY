import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation, ThomasFermi
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 21,   
    'ytick.labelsize': 21,    
    'legend.fontsize': 23,    
    'axes.grid': False        
})

def graf(L, density, phase, file):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    zoom_region = [-L[0]/8, L[0]/8, -L[0]/8, L[0]/8]

    im0 = ax[0].imshow(density, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[0].set_title(r"$|\phi|^2$", size=23)
    ax[0].axis(zoom_region)
    ax[0].set_xticks([])   
    ax[0].set_yticks([])

    im1 = ax[1].imshow(phase, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='Greys')
    ax[1].set_title(r"$S(\rho)$", size=23)
    ax[1].axis(zoom_region)
    ax[1].set_xticks([])   
    ax[1].set_yticks([])

    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])

    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/{file}_anisotropia.png', dpi=300)

def nucleazioa():
    beta = 100.0    
    gamma = (10.0, 10.1)
    sigma = (1.0,1.0)
    Omega = 0.7
    tf = ThomasFermi(gamma, beta)
    N = (2**8, 2**8)
    L = (10*tf.rtf, 10*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L, L[0]/N[0])
    n_vortex = 0
    tol = 1e-10
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

    graf(L, sim.wf.density(), sim.wf.phase(), "hasiera")
    
    phi_0_ruta = f"../saves/phi{round(Omega,1)}_{round(gamma[0],1)}-{round(gamma[1],1)}_{round(sigma[0],1)}-{round(sigma[1],1)}_{n_vortex}_{tol:.0e}_{N[0]}-{N[1]}_{round(L[0],3)}_{int(beta)}.npy"
    if os.path.exists(phi_0_ruta):
        print(f"Cargando estado fundamental desde {phi_0_ruta}...")
        sim.wf.phi = np.load(phi_0_ruta)
        print("Estado fundamental cargado.")
    else:
        print("No se ha encontrado estado fundamental precargado.\nIniciando proceso de cooling (evolucion imaginaria)...")
        sim.cooling(1e-4, tol=tol, max_iter=20000000)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Nuevo estado fundamental guardado en {phi_0_ruta}")

    graf(L, sim.wf.density(), sim.wf.phase(), "bukaera")

if __name__ == "__main__":
    nucleazioa()