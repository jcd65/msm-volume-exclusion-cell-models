import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import get_cmap
from scipy.integrate import solve_ivp

def myopic_exp(t,C, rProlif):
    """Myopic Exponential ODE with some rate of proliferation."""
    dCdt = np.zeros(1)
    dCdt[0] = rProlif*C*(1-C**4)
    return dCdt

def logistic_exp(t,C, rProlif):
    """Exponential ODE with some rate of prolfieration."""
    dCdt = np.zeros(1)
    dCdt[0] = rProlif*C*(1-C)
    return dCdt

def reset_ODEs(t,C, rProgress, K):
    """Reset ODE with some rate of progression and K stages."""
    dCdt = np.zeros(K+1)
    #Proliferation/progression into first stage
    dCdt[0] = rProgress*C[-2]*(1-C[-1]) + rProgress*C[-2] - rProgress*C[0]

    #Progression through intermediate stages
    for s in range(1,len(C)-1):
        dCdt[s] = rProgress*C[s-1] - rProgress*C[s]
    
    #Total population
    dCdt[-1] = rProgress*C[-2]*(1-C[-1])

    return dCdt

def remain_ODEs(t,C, rProgress, K):
    """Remain ODE with some rate of progression and K stages."""
    dCdt = np.zeros(K+1)
    #Proliferation/progression into first stage
    dCdt[0] = 2*rProgress*C[-2]*(1-C[-1]) - rProgress*C[0]

    #Progression through intermediate stages
    for s in range(1,len(C)-2):
        dCdt[s] = rProgress*C[s-1] - rProgress*C[s]
    
    #Stage K cells
    dCdt[-2] = rProgress*C[-3] - rProgress*C[-2] * (1-C[-1])

    #Total population
    dCdt[-1] = rProgress*C[-2]*(1-C[-1])

    return dCdt

def reset_myopic(t,C, rProgress, K):
    """Myopic Reset ODE with some rate of progression and K stages."""
    dCdt = np.zeros(K+1)

    #Proliferation/progression into first stage
    dCdt[0] = rProgress*C[-2]*(1-(C[-1])**4) + rProgress*C[-2] - rProgress*C[0]

    #Progression through intermediate stages
    for s in range(1,len(C)-1):
        dCdt[s] = rProgress*C[s-1] - rProgress*C[s]
    
    #Total population
    dCdt[-1] = rProgress*C[-2]*(1-(C[-1])**4)

    return dCdt

def remain_myopic(t,C, rProgress, K):
    """"Myopic Remain ODE with some rate of progression and K stages."""
    dCdt = np.zeros(K+1)

    #Proliferation/progression into first stage
    dCdt[0] = 2*rProgress*C[-2]*(1-(C[-1])**4) - rProgress*C[0]

    #Progression through intermediate stages
    for s in range(1,len(C)-2):
        dCdt[s] = rProgress*C[s-1] - rProgress*C[s]
    
    #Stage K cells
    dCdt[-2] = rProgress*C[-3] - rProgress*C[-2] * (1-(C[-1])**4)

    #Total population
    dCdt[-1] = rProgress*C[-2]*(1-(C[-1])**4)

    return dCdt

def ODE_solver(t_init, t_final, dt,C0, rProlif, ODE_func):
    """Solve a given ODE for an Exponential model given initial condition and proliferation rate.
        :param t_init: Initial time
        :param t_final: Final time
        :param dt: Time step
        :param C0: Initial condition
        :param rProlif: Proliferation rate
        :param ODE_func: The ODE function to solve
    """
    num_steps = int((t_final - t_init) / dt)
    t_eval = np.linspace(t_init, t_final, num_steps)
    sol = solve_ivp(ODE_func, [t_init, t_final], C0, args=(rProlif,), t_eval=t_eval)
    return sol.t, sol.y

def ODE_solver_multistage(t_init, t_final, dt, C0, rProgress, K, ODE_func):
    """Solve the ODE for a cell proliferation model given initial condition, progression rate, and number of stages.
        :param t_init: Initial time
        :param t_final: Final time
        :param dt: Time step
        :param C0: Initial condition
        :param rProgress: Progression rate (remember to scale if working with a prolfieration rate)
        :param K: Number of stages
        :param ODE_func: The ODE function to solve
    """
    num_steps = int((t_final - t_init) / dt)
    t_eval = np.linspace(t_init, t_final, num_steps)
    sol = solve_ivp(ODE_func, [t_init, t_final], C0, args=(rProgress, K), t_eval=t_eval)
    return sol.t, sol.y