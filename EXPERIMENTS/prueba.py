import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation, ThomasFermi
import scienceplots
plt.style.use(['science', 'ieee'])

def test():
    beta = 1000.0
    gamma = (10.0, 10.0)
    Omega = 0.9
    tf = ThomasFermi(gamma, beta)
    N = (2**8, 2**8)
    L = (8*tf.rtf, 8*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L)
    n_vortex = 1
    vortex_charge = [1]
    positions = [
        (1.0, 0.0)
    ]
    t=1
    tol=1e-9

    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = n_vortex, 
                     vortex_charge = vortex_charge, 
                     positions     = positions
                     )
    
    phi_0_ruta = f"../saves/phi{round(Omega,1)}_{round(gamma[0],1)}-{round(gamma[1],1)}_{n_vortex}_{tol:.0e}_{N[0]}-{N[1]}_{int(beta)}.npy"
    if os.path.exists(phi_0_ruta):
        print(f"Cargando estado fundamental desde {phi_0_ruta}...")
        sim.wf.phi = np.load(phi_0_ruta)
        print("Estado fundamental cargado.")
    else:
        print("No se ha encontrado estado fundamental precargado.\nIniciando proceso de cooling (Gradient descent)...")
        sim.cooling(1e-4, tol=tol)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Nuevo estado fundamental guardado en {phi_0_ruta}")

    # 5. Vamos a simular la hidrodinamica
    sim.hydrodynamics(5.0,dt=dt)
    density5 = sim.wf.density()
    print(5)

    sim.hydrodynamics(5.0,dt=dt)
    density10 = sim.wf.density()
    print(10)

    sim.hydrodynamics(5.0,dt=dt)
    density15 = sim.wf.density()
    print(15)

    sim.hydrodynamics(5.0,dt=dt)
    density20 = sim.wf.density()
    print(20)

    sim.hydrodynamics(5.0,dt=dt)
    density25 = sim.wf.density()
    print(25)

    # 6. Visualización de resultados
    fig, ax = plt.subplots(2, 3, figsize=(12, 5))
    zoom_region = [-sim.rtf, sim.rtf, -sim.rtf, sim.rtf]

    im0 = ax[0,0].imshow(density0, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[0,0].set_title(r"$|\Psi|^2 (t=0s)$")
    ax[0,0].axis(zoom_region)
    fig.colorbar(im0, ax=ax[0,0])

    im1 = ax[0,1].imshow(density5, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[0,1].set_title(r"$|\Psi|^2 (t=5s)$")
    ax[0,1].axis(zoom_region)
    fig.colorbar(im1, ax=ax[0,1])

    im2 = ax[0,2].imshow(density10, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[0,2].set_title(r"$|\Psi|^2 (t=10s)$")
    ax[0,2].axis(zoom_region)
    fig.colorbar(im2, ax=ax[0,2])

    im3 = ax[1,0].imshow(density15, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[1,0].set_title(r"$|\Psi|^2 (t=15s)$")
    ax[1,0].axis(zoom_region)
    fig.colorbar(im3, ax=ax[1,0])

    im4 = ax[1,1].imshow(density20, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[1,1].set_title(r"$|\Psi|^2 (t=20s)$")
    ax[1,1].axis(zoom_region)
    fig.colorbar(im4, ax=ax[1,1])

    im5 = ax[1,2].imshow(density25, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[1,2].set_title(r"$|\Psi|^2 (t=25s)$")
    ax[1,2].axis(zoom_region)
    fig.colorbar(im5, ax=ax[1,2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()