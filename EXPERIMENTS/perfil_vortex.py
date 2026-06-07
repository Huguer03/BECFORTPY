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
    'legend.fontsize': 20,    
    'axes.grid': True        
})

def graf(rtf,x_axis,profile_sim,profile_tf,Omega,beta):
    plt.figure(figsize=(8, 6))
    zoom = [0, 2.5]
    plt.plot(x_axis, profile_sim, label="Bortize profila", linewidth=2)
    plt.plot(x_axis, profile_tf, linestyle='--', label="Bortize profilaren hurbilketa", linewidth=2)
    plt.xlabel(r"$\rho$", size=30)
    plt.ylabel(r"$|\phi|^2$", size=30)
    plt.xlim(zoom)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/profila_{str(round(Omega,1)).replace('.', '-')}_{int(beta)}_bortize_hidrodinamika.png', dpi=300)

def test_vortex_central():
    beta = 500.0
    gamma = (10.0, 10.0)
    Omega = 0.4
    tf = ThomasFermi(gamma, beta)
    N = (2**8, 2**8)
    L = (10*tf.rtf, 10*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L)
    n_vortex = 1
    vortex_charge = [1]
    positions = [
        (0.0, 0.0)
    ]
    t=3
    tol=1e-13

    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
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
        sim.cooling(5e-4, tol=tol, max_iter=20000000)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Nuevo estado fundamental guardado en {phi_0_ruta}") 

    sim.hydrodynamics(t_max=t,dt=1e-3)
    density_sim = sim.wf.density() 

    r2 = grid.X**2 + grid.Y**2
    r  = np.sqrt(r2)
    density_tf = gamma[0]**2 * (tf.rtf**2 - r2) / (2 * beta)
    density_tf[density_tf < 0] = 0

    rho_0 = density_tf.max()
    xi = 1.0 / np.sqrt(2 * beta * rho_0)
    f_s = r / np.sqrt(xi**2 + r2)

    density_tf = density_tf * (f_s**2)
    density_tf[density_tf < 0] = 0 
    
    mid_idx = N[1] // 2
    x_axis = grid.x[mid_idx:]
    profile_sim = density_sim[:, mid_idx]
    profile_tf = density_tf[:, mid_idx]

    graf(tf.rtf,grid.x,profile_sim,profile_tf,Omega,beta)

if __name__ == "__main__":
    test_vortex_central()