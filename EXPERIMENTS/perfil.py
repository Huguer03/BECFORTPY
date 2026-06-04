import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation, ThomasFermi
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 27,   
    'ytick.labelsize': 27,    
    'legend.fontsize': 23,    
    'axes.grid': True        
})

def graf(rtf,x_axis,diff_tf,profile_sim,profile_tf,Omega,beta):
    plt.figure(figsize=(8, 6))
    zoom = [0, (rtf + 1.5)]

    plt.plot(x_axis, profile_sim, label="Simulazioa", lw=1.5)
    plt.plot(x_axis, profile_tf, label="Thomas-Fermi", color='black', lw=1.5)
    plt.axvline(x=rtf, linestyle='--', alpha=0.7, label=r'$\bar{R}_{\text{TF}}$')
    
    plt.xlabel(r"$\rho$", size=30)
    plt.ylabel(r"$|\phi|^2$", size=30)
    plt.xlim(zoom)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/profila_{str(round(Omega,1)).replace('.', '-')}_{int(beta)}.png', dpi=300)

def test_comparacion_tf(Omega):
    beta = 1000.0
    gamma = (10.0, 10.0)
    Omega = Omega
    tf = ThomasFermi(gamma, beta)
    rtf, mutf = tf.rtf, tf.mutf
    N = (2**8, 2**8)
    L = (10*rtf, 10*rtf)
    grid = Grid(N, L)
    print(rtf, L)
    n_vortex = 0
    tol = 1e-13
    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = n_vortex, 
                     vortex_charge = None, 
                     positions     = None
                     )

    phi_0_ruta = f"../saves/phi{round(Omega,1)}_{round(gamma[0],1)}-{round(gamma[1],1)}_{n_vortex}_{tol:.0e}_{N[0]}-{N[1]}_{round(L[0],3)}_{int(beta)}.npy"
    if os.path.exists(phi_0_ruta):
        print(f"Cargando estado fundamental desde {phi_0_ruta}...")
        sim.wf.phi = np.load(phi_0_ruta)
        print("Estado fundamental cargado.")
    else:
        print("No se ha encontrado estado fundamental precargado.\nIniciando proceso de cooling (Gradient descent)...")
        sim.cooling(1e-5, tol=tol)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Nuevo estado fundamental guardado en {phi_0_ruta}")
    density_sim = sim.wf.density()

    r2 = grid.X**2 + grid.Y**2
    density_tf = gamma[0]**2 * (rtf**2 - r2) / (2 * beta)
    density_tf[density_tf < 0] = 0
    
    mid_idx = N[1] // 2
    x_axis = grid.x
    profile_sim = density_sim[:, mid_idx]
    profile_tf = density_tf[:, mid_idx]

    diff_tf = np.abs(profile_tf - profile_sim)

    graf(rtf,x_axis,diff_tf,profile_sim,profile_tf,Omega,beta)

if __name__ == "__main__":
    Omega = [0.0,0.5,0.8,0.9]
    for i in Omega:
        print(i)
        test_comparacion_tf(i)