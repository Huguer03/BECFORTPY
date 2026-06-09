import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 27,   
    'ytick.labelsize': 27,    
    'legend.fontsize': 20,    
    'axes.grid': True        
})

def vortex_ode(eta, y, s=1):
    f, g = y
    if eta < 1e-12:
        df = g
        dg = 0.0
    else:
        df = g
        dg = -g/eta - (1 - s*s/(eta*eta))*f + f**3
    return [df, dg]

def shoot(c, eta_max=15, eta0=1e-6, s=1):
    f0 = c * eta0       
    g0 = c               
    sol = solve_ivp(vortex_ode, [eta0, eta_max], [f0, g0], args=(s,),
                    method='LSODA', rtol=1e-13, atol=1e-10)
        return 0.5
    return sol.y[0, -1]

c_low, c_high = 0.5, 0.7
target = 1.0
for i in range(25):
    c_mid = (c_low + c_high) / 2
    f_end = shoot(c_mid)
    if f_end < target:
        c_low = c_mid
    else:
        c_high = c_mid
c_opt = (c_low + c_high) / 2

eta0, eta_max = 1e-6, 10
sol_final = solve_ivp(vortex_ode, [eta0, eta_max], [c_opt*eta0, c_opt],
                      args=(1,), method='LSODA', rtol=1e-13, atol=1e-12,
                      dense_output=True)
eta_vals = np.linspace(eta0, eta_max, 2000)
f_vals = sol_final.sol(eta_vals)[0]

np.savez("../saves/perfil_vortex.npz", eta=eta_vals, f=f_vals)

plt.figure(figsize=(8, 6))
plt.plot(eta_vals, f_vals, label='Perfil numérico (solve_ivp)', linewidth=2)
plt.axhline(1, color='gray', linestyle='--', label='asíntota')
plt.xlabel(r'$\eta = r/\xi$', size=23)
plt.ylabel(r'$f(\eta)$', size=23)
plt.xlim([0,10])
plt.ylim([0,1.1])
plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/f_eta.png', dpi=300)