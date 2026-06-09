import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, Simulation, ThomasFermi


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
    t=0.2

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
        sim.cooling(1e-3, tol=tol, max_iter=20000000)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Nuevo estado fundamental guardado en {phi_0_ruta}")

    print(sim.wf.norma())
    density0 = sim.wf.density()
    #w = np.exp(-4*(grid.X**2 + grid.Y**2)/7)
    # 5. Vamos a simular la hidrodinamica
    #sim.hydrodynamics(t_max=t,dt=1e-3, gamma=(0.0,0.0), Omega=0.0)
    density0 = sim.wf.phase()
    density5 = sim.wf.density()
    print(sim.wf.vortices(rmax=tf.rtf+0.5))
    print(t)

    # 6. Visualización de resultados
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    zoom_region = [-4*tf.rtf, 4*tf.rtf, -4*tf.rtf, 4*tf.rtf]

    im0 = ax[0].imshow(density0, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[0].set_title(r"$|\Psi|^2 (t=0s)$")
    ax[0].axis(zoom_region)
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(density5, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[1].set_title(r"$|\Psi|^2 (t={}s)$".format(t))
    ax[1].axis(zoom_region)
    fig.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()