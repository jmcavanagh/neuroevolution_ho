

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
# a = average, y/p is pos/momentum, 1,2,3 is exp. so, ay2 is <y**2>



def deriv(t, y, traj_beta, traj_k, times):
    beta = np.interp(t, times, traj_beta)
    k = np.interp(t, times, traj_k)
    ay, ap, apy, ay2, ap2 = y
    day_dt = ap / m
    dap_dt = -zeta / m * ap - k*ay
    dapy_dt = ap2 / m - k*ay2 - zeta / m * apy
    day2_dt = 2 * apy / m
    dap2_dt = -2*k*apy - 2*zeta * ap2 + 2*zeta / beta
    return [day_dt, dap_dt, dapy_dt, day2_dt, dap2_dt]
    

#calculates excess work
def excess_work(traj_beta, traj_k, times):
    ay_0 = 0
    ap_0 = 0
    ay2_0 =  1/(traj_k[0] * traj_beta[0])
    apy_0 = 0
    ap2_0 = m / traj_beta[0]
    y0 = [ay_0, ap_0, apy_0, ay2_0, ap2_0]
    sol = solve_ivp(fun=lambda t, y:deriv(t,y,traj_beta, traj_k, times), t_span=(0,100), y0=y0)
    t = sol.t
    ay = sol.y[0,:]
    ap = sol.y[1,:]
    apy = sol.y[2,:]
    ay2 = sol.y[3,:]
    ap2 = sol.y[4,:]
    beta = np.interp(t, times, traj_beta)
    k = np.interp(t, times, traj_k)
    beta_dot = np.gradient(beta, t)
    k_dot = np.gradient(k, t)
    integrand = beta_dot*ap2 / (2*m) + 0.5*ay2*(k*beta_dot + k_dot*beta) - beta_dot / beta - k_dot / (2*k)
    return np.trapz(integrand, t)




#problem params
m = 1
zeta = 1
times = np.array(range(0, 100))


#Example beta and k
traj_beta = np.linspace(4, 10, 100)
traj_k = np.linspace(60, 15, 100)

#initial values


if __name__ == '__main__':
    xsw = excess_work(traj_beta, traj_k, times)
    print(xsw)





