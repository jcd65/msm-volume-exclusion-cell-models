import numpy as np
import random as rand

def simulate_myopic_prolif_new(init_latt,t_init,t_final,dt,rp,rm,K,k_reset):
    """
    This function moves and proliferates cells from t_init to t_final according to the methods described in the paper.
    This is done by generating a time to next reaction, and choosing a reaction as in the Gillespie algorithm.
    When a cell chooses to proliferate, it will choose (uniformly at random) between empty sites if there are any available.
    If a particle fails to proliferate, it is reset to stage k_reset.
    The lattice and positions are then updated acoording to the chosen reaction, and this process repeats
    until time T_final is reached/exceeded. 

    :param init_latt: Initial lattice with particles, array
    :param init_pos: Initial positions of particles, list
    :param t_init: Initial time, float
    :param t_final: Final time, float
    :param rp: Rate of progression through stages, float
    :param rm: Rate of movement, float
    :param dt: Time step for simulation, float
    :param K: Number of stages, int
    :param k_reset: Stage to reset to if cell proliferation fails, int
    :return: Updated lattice , final time, array of number of attempted and failed proliferation attempts
    """
   #Get Lattice dimension
    Lx, Ly = np.shape(init_latt)

    #number of timesteps
    num_timesteps = int((t_final - t_init) / dt)

    #Initialise lattice and number of particles array
    lattices = np.zeros((num_timesteps+1, Lx, Ly))
    lattices[0] = np.copy(init_latt)

    #Initialise time, keep track of current time
    t_now = t_init
    checkpoint = t_init

    #Initialise lattice and positions
    lattice_now = np.copy(init_latt)

    #Initialise failed proliferation and proliferation attempts counter
    num_failed_prolif_events = 0
    num_prolif_events = 0
    failed_prolifs_array = np.zeros(num_timesteps+1)
    prolif_attempts_array = np.zeros(num_timesteps+1)

    #Initialise propensities (K+1 of them as there is propoensity for movement as well)
    a = np.zeros(2)

    #propensity for movement
    a[0] = rm*np.count_nonzero(init_latt)

    #propensity for stage progression
    a[1] = rp*np.count_nonzero(init_latt)


    #Loop until final times
    while t_now < t_final:
        #Break if no cells left
        if a[1] == 0: #so zero propensity
            break

        lattice_old = np.copy(lattice_now)
        t_old = t_now

        #cumulative sums of propensities
        cumsuma = np.cumsum(a)
        
        #sum of propoensities
        a0 = cumsuma[-1]

        #Generate time to next reaction
        tau = (1/a0)*np.log(1/rand.random())

        #Choose which reaction occurs
        ra0 = a0*rand.random()
        
        #Choose a random nonzero lattice position for the reaction to occur
        idx = np.random.choice(np.nonzero(lattice_now.flatten())[0])
    
        #Convert index to 2D coordinates
        x_idx = idx // Lx
        y_idx = idx % Lx

        #Movement event
        if ra0 < cumsuma[0]:
            #Choose a random direction to move
            directions = ["up", "down", "left", "right"]
            direction = np.random.choice(directions)

            #get stage of cell
            cell_stage = int(lattice_old[x_idx, y_idx])

            #Attempt to move the cell
            if direction == "up":
                new_x = x_idx
                new_y = (y_idx + 1) % Ly
                if lattice_now[new_x, new_y] == 0:  # Check if new position is empty
                    lattice_now[x_idx, y_idx] = 0  # Clear old position
                    lattice_now[new_x, new_y] = cell_stage  # Set new position
            elif direction == "down":
                new_x = x_idx
                new_y = (y_idx - 1) % Ly
                if lattice_now[new_x, new_y] == 0:  # Check if new position is empty
                    lattice_now[x_idx, y_idx] = 0  # Clear old position
                    lattice_now[new_x, new_y] = cell_stage  # Set new position
            elif direction == "left":
                new_x = (x_idx - 1) % Lx
                new_y = y_idx
                if lattice_now[new_x, new_y] == 0:  # Check if new position is empty
                    lattice_now[x_idx, y_idx] = 0  # Clear old position
                    lattice_now[new_x, new_y] = cell_stage  # Set new position
            else:  # direction == "right"
                new_x = (x_idx + 1) % Lx
                new_y = y_idx
                if lattice_now[new_x, new_y] == 0:  # Check if new position is empty
                    lattice_now[x_idx, y_idx] = 0  # Clear old position
                    lattice_now[new_x, new_y] = cell_stage  # Set new position        
            #Note no need to update propensities here, as it is the same as before
        
        #Proliferation or Progression event
        else:
            #Progress stage if not in last stage
            if lattice_old[x_idx,y_idx] < K:
                #Progress stage
                lattice_now[x_idx, y_idx] += 1  # Increment stage
                #Note no need to update propensities here, as it is the same as before

            #Otherwise attempt proliferation
            else:
                #Increment proliferation attempts counter
                num_prolif_events += 1

                #Attempt proliferation
                success = False


                #Choose a random direction to proliferate
                directions = ["up", "down", "left", "right"]
                #randomly shuffle the directions
                rand.shuffle(directions)

                for direction in directions:
                    #Attempt to proliferate the cell
                    if direction == "up":
                        new_x = x_idx
                        new_y = (y_idx + 1) % Ly
                        if lattice_old[new_x, new_y] == 0:  # Check if new position is empty
                            lattice_now[new_x, new_y] = 1  # add new cell
                            lattice_now[x_idx, y_idx] = 1  # reset old cell
                            success = True  # Proliferation successful
                            break #Stop trying new directions in directions array
                    
                    elif direction == "down":
                        new_x = x_idx
                        new_y = (y_idx - 1) % Ly
                        if lattice_old[new_x, new_y] == 0:  # Check if new position is empty
                            lattice_now[new_x, new_y] = 1  # add new cell
                            lattice_now[x_idx, y_idx] = 1  # reset old cell
                            success = True  # Proliferation successful
                            break #Stop trying new directions in directions array
                    
                    elif direction == "left":
                        new_x = (x_idx - 1) % Lx
                        new_y = y_idx
                        if lattice_old[new_x, new_y] == 0:  # Check if new position is empty
                            lattice_now[new_x, new_y] = 1  # add new cell
                            lattice_now[x_idx, y_idx] = 1  # reset old cell
                            success = True  # Proliferation successful
                            break #Stop trying new directions in directions array
                    
                    else:  # direction == "right"
                        new_x = (x_idx + 1) % Lx
                        new_y = y_idx
                        if lattice_old[new_x, new_y] == 0:  # Check if new position is empty
                            lattice_now[new_x, new_y] = 1  # add new cell
                            lattice_now[x_idx, y_idx] = 1  # reset old cell
                            success = True  # Proliferation successful 
                            break #Stop trying new directions in directions array   

                #Only update propensities if proliferation was successful
                if success:
                    #Add by rate as we can only increase by one cell at a time
                    a[0] += rm
                    a[1] += rp
                else:
                    #If proliferation fails, reset to stage k_reset, increment failed attempts counter
                    lattice_now[x_idx, y_idx] = k_reset
                    num_failed_prolif_events += 1

        #Update time
        t_now += tau

        # Calculate the indices of the time step before and the current time
        # step in terms of recording
        ind_before = int(np.ceil((t_old+ np.finfo(float).eps) / dt))
        ind_after = min(int(np.floor(t_now/ dt)), num_timesteps)
        
        # Find out how many time-steps to write to
        steps_to_write = ind_after - ind_before + 1
        
        if steps_to_write > 0 and steps_to_write != float('inf'):
            for i in range(ind_before, ind_after + 1):
                lattices[i,:,:] = np.copy(lattice_now)
                failed_prolifs_array[i] = num_failed_prolif_events
                prolif_attempts_array[i] = num_prolif_events

        #break if lattice is full
        if a[0]/rm >= Lx*Ly:
            #Fill remaining time steps with the current lattice
            for i in range(ind_after, num_timesteps + 1):
                lattices[i,:,:] = np.copy(lattice_now)
            break
    return lattices, t_now, failed_prolifs_array, prolif_attempts_array