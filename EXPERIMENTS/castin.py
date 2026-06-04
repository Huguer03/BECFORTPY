import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from becFort import Grid, Simulation, ThomasFermi, TrapPotential
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 16,   
    'ytick.labelsize': 16,    
    'legend.fontsize': 14,    
    'axes.grid': True        
})

def ecuaciones_castin_dum(t, y, omega_x, omega_y):
    bx, dbx, by, dby = y
    
    ddbx = (omega_x**2) / (bx**2 * by)
    ddby = (omega_y**2) / (bx * by**2)
    
    return [dbx, ddbx, dby, ddby]

def calcular_radio_rms(X, Y, densidad):
    norma = np.sum(densidad)
    x_rms = np.sqrt(np.sum(X**2 * densidad) / norma)
    y_rms = np.sqrt(np.sum(Y**2 * densidad) / norma)
    return x_rms, y_rms

def castin():
    beta = 1000.0
    gamma = (10.0, 20.0)
    Omega = 0
    tf = ThomasFermi(gamma, Omega, beta)
    N = (2**8, 2**8)
    L = (8*tf.rtf, 8*tf.rtf)
    grid = Grid(N, L)
    dt = 1e-4
    t_max = 0.2
    pasos_dinamica = int(t_max / dt)

    print("1. Encontrando el estado fundamental (Cooling)...")
    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = 0, 
                     vortex_charge = None, 
                     positions     = None
                     )
    sim.cooling(dt=1e-4, tol=1e-6)

    densidad_inicial = sim.wf.density()
    r_x0_num, r_y0_num = calcular_radio_rms(grid.X, grid.Y, densidad_inicial)

    V = np.zeros_like(grid.X) 

    tiempos_num = np.linspace(0, t_max, 11)
    bx_num = [1.0]
    by_num = [1.0]

    t_actual = 0.0
    dt_dinamica = t_max / 10.0

    for i in range(10):
        sim.hydrodynamics(t_max=dt_dinamica, dt=dt, V=V)
        t_actual += dt_dinamica
        
        densidad_t = sim.wf.density()
        rx_t, ry_t = calcular_radio_rms(grid.X, grid.Y, densidad_t)
        
        bx_num.append(rx_t / r_x0_num)
        by_num.append(ry_t / r_y0_num)

    y0 = [1.0, 0.0, 1.0, 0.0]
    tiempos_teo = np.linspace(0, t_max, 200)

    sol = solve_ivp(ecuaciones_castin_dum, [0, t_max], y0, t_eval=tiempos_teo, args=(gamma[0], gamma[1]))
    bx_teo = sol.y[0]
    by_teo = sol.y[2]

    plt.figure(figsize=(8, 5))
    plt.plot(tiempos_teo, bx_teo, 'b-', label=r'$b_x(t)$ (Castin-Dum teorikoa)')
    plt.plot(tiempos_teo, by_teo, 'r-', label=r'$b_y(t)$ (Castin-Dum teorikoa)')
    plt.plot(tiempos_num, bx_num, 'bo', label=r'$b_x(t)$ (Simulazioa)')
    plt.plot(tiempos_num, by_num, 'ro', label=r'$b_y(t)$ (Simulazioa)')

    plt.xlabel(r'$\tau (\omega_{ho}$)', size=18)
    plt.ylabel(r'Eskala faktoreak $b_i(\tau)$', size=18)
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/test_castin_dum.png', dpi=300)

if __name__ == "__main__":
    castin()