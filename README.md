## Code files
The files [prolif_periodic.py](prolif_periodic.py), [prolif_reflecting.py](prolif_reflecting.py), [myopic_periodic.py](myopic_periodic.py), and [myopic_reflecting.py](myopic_reflecting.py) contain functions which can be used to run ABMs with different boundary behaviour ona rectangular domain. Use prolif files for Non-Myopic models and myopic files for Myopic models. Periodic files impose periodic boundaries, whilst reflecting files impose reflecting vertical boundaries and periodic horizontal boundaries. ABM functions output the lattice evolution over time, the final time of the simulation, an array of failed proliferation attempts over time, and an array of the number of proliferation attempts over time.

An example implementation of the Remain model with $K=10$ stages and $r_p = r_m=1$ on a $100 \times 100$ lattice is given below:
```python
from prolif_periodic import simulate_prolif
import numpy as np
import random as rand
Lx = 100
Ly = 100
rp = 1
rm = 1
K = 10
T_Init = 0
T_final = 10
dt = 0.1
k_reset = 10
init_lattice = np.zeros((Lx,Ly)) #Fill lattice as desired
lattices, t_now, failed_prolifs_array, prolif_attempts_array = simulate_prolif(init_lattice,T_init,T_final,dt,rp,rm,K,k_reset)
```

The file [PCF.py](PCF.py) can be used to calculate the PCF of a lattice. Import all functions from [PCF.py](PCF.py) and call the `PCF` to calculate the PCF of a lattice at some distance. See below, an example of an implementation of the PCF function:
```Python
import numpy as np
import PCF

Lattice = np.zeros((100,100)) #Use desired lattice here
distance = 1 #Change distance as necessary
pcf = PCF(Lattice, distance)
```

The files [ODEs.py](ODEs.py) and [PDEs.py](PDEs.py) solve the continuum models in the case where our uniform initial conditions are translationally invariant and translationally invariant in the y-axis. From [ODEs.py](ODEs.py), call `ODE_solver` to solve the Exponential model ODEs or call `ODE_solver_myopic` to solve the multi-stage models ODEs. From [PDEs.py](PDEs.py) call `PDE_solver`. For example, to solve the Myopic Remain PDEs, implement the following code:

```Python
import numpy as np
from scipy.integrate import solve_ivp
from PDEs import d2_neumann_axis0
from PDEs import myopic_remain_PDE
from PDEs import PDE_solver

t_init = 0
t_final = 300
dt = 0.5
Nx = 800
dx = 0.25
K = 10
r = 1
D = 1
C0 = np.zeros(Nx) #Replace with desired initial condition
PDE_sol = PDE_solver(t_init, t_final, dt, C0, K, Nx, r, D, dx, myopic_remain_PDE)
```

The file [wavespeeds.py](wavespeeds.py) is used to calculate the wave-speed of a travelling wavefront given an array of the form $(\text{times}, \text{concentrations})$. Call the `wavespeed` function to calculate the wavespeed at each timepoint of a travelling wave. Call `moving_average` to calculate the average wave-speed of travelling wave over some time window. See below for example implementation of both functions:
```Python
import numpy as np
from wavespeeds import wavespeed
from wavespeeds import moving_average

dt = 0.5
T_init = 0
T_final = 300
timesteps = int((T_final - T_init)/dt)
Lx = 300
Concentration_evol = np.zeros((timesteps, Lx)) #Fill with desired concentration

#Wavespeeds per unit time
wavespeeds = wavespeed(Concentration_evol, dt)

#Moving average wave-speed
window_size = 5
moving_average_wavespeed = moving_average(wavespeeds, window_size)
```



