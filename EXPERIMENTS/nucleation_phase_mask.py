import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, Simulation, ThomasFermi
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
    ax.set_title(rf"$\tau = {t}$", size=23)
    ax.axis(zoom_region)
    ax.set_xticks([])   
    ax.set_yticks([])

    fig.colorbar(im, ax=ax)

    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/fase_mascara_w/{str(t)}_densidad.png', dpi=300)
    plt.close()

def graf_phase(L, phase, t):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
    zoom_region = [-L[0]/6, L[0]/6, -L[0]/6, L[0]/6]

    im = ax.imshow(phase, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='Greys')
    ax.set_title(rf"$\tau = {t}$", size=23)
    ax.axis(zoom_region)
    ax.set_xticks([])   
    ax.set_yticks([])

    fig.colorbar(im, ax=ax)

    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/fase_mascara_w/{str(t)}_phase.png', dpi=300)
    plt.close()

def nucleazioa():
    beta = 500.0    
    gamma = (10.0, 10.0)
    sigma = (1.0,1.0)
    Omega = 0.2
    tf = ThomasFermi(gamma, beta)
    N = (2**8, 2**8)
    L = (10*tf.rtf, 10*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L, L[0]/N[0])
    n_vortex = 0
    tol = 1e-13
    vortex_charge = [1]
    positions = [
        (0.0, 0.0)
    ]
    t  = np.linspace(1,100,100)
    dt = t[1]-t[0]

    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     sigma         = sigma,
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = n_vortex, 
                     vortex_charge = vortex_charge, 
                     positions     = positions
                     )
    
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

    phase = np.atan2(grid.Y, grid.X)
    
    vortex_phase = np.exp(1j * phase)
    
    sim.wf.phi *= vortex_phase

    graf_densidad(L, sim.wf.density(), 0)
    graf_phase(L, sim.wf.phase(), 0)
    w = np.full_like(grid.X, 0.03)
    for i in t:
        sim.hydrodynamics(t_max=dt,dt=1e-3, gamma=(10.0,10.0), diss=True, W=w)
        graf_densidad(L, sim.wf.density(), i)
        graf_phase(L, sim.wf.phase(), i)

if __name__ == "__main__":
    nucleazioa()