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

def graf(v,o,b):
    plt.figure(figsize=(8, 6))
    plt.plot(v[0], v[2], label="Simulazioa", linewidth=2)
    plt.plot(v[0], v[1], linestyle='--', label="Thomas-Fermi hurbilketa", linewidth=2)
    plt.axvline(v[3], linestyle='--', label=r'$\bar{R}_{\text{TR}}$')
    plt.xlim([0,3])
    plt.grid(True)
    if o == 0.0:
        plt.ylabel(r'$|\phi|^2$', size=27)
    if b == 1000:
        plt.xlabel(fr'$\rho$', size=27)
    if o == 0.8 and b == 10:
        plt.legend()
    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/profila_{str(o).replace('.', '-')}_{b}.png', dpi=300)


def test_comparacion_tf(Omega, beta):
    beta = beta
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

    phi_0_ruta = f"../saves/phi{round(Omega,1)}_{round(gamma[0],1)}-{round(gamma[1],1)}_{round(sigma[0],1)}-{round(sigma[1],1)}_{n_vortex}_{tol:.0e}_{N[0]}-{N[1]}_{round(L[0],3)}_{int(beta)}.npy"
    if os.path.exists(phi_0_ruta):
        print(f"Cargando estado fundamental desde {phi_0_ruta}...")
        sim.wf.phi = np.load(phi_0_ruta)
        print("Estado fundamental cargado.")
    else:
        print("No se ha encontrado estado fundamental precargado.\nIniciando proceso de cooling (evolucion imaginaria)...")
        sim.cooling(5e-4, tol=tol, max_iter=20000000)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Nuevo estado fundamental guardado en {phi_0_ruta}")

    r2 = grid.X**2 + grid.Y**2
    density_tf = gamma[0]**2 * (rtf**2 - r2) / (2 * beta)
    density_tf[density_tf < 0] = 0
    
    mid_idx = N[1] // 2
    x_axis = grid.x
    profile_sim = density_sim[:, mid_idx]
    profile_tf = density_tf[:, mid_idx]

    return x_axis.copy(),profile_tf.copy(),profile_sim.copy(), rtf

if __name__ == "__main__":
    Omega = [0.0, 0.5, 0.8]
    Beta = [10, 100, 1000]
    for b in Beta:
        for o in Omega:
            print(f"Simulando para beta={b}, omega={o}...")
            v = test_comparacion_tf(o, b)
            graf(v,round(o,1),int(b))