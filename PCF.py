import numpy as np

def offset_L1(m: int):
    """Calculate the offsets required for distance m

    :param m: The distance for which to calculate offsets

    :return: A list of offsets for the given distance
    """

    offsets = []
    for dx in range(m+1):
        dy = m-dx
        if dx == 0:
            offsets += [(0,dy), (0,-dy)]
        elif dy == 0:
            offsets += [(dx,0), (-dx,0)]
        else:
            offsets += [(dx,dy), (dx,-dy), (-dx,dy), (-dx,-dy)]
    return offsets

def mdist_neighbours(lattice: np.ndarray, m: int, idx: tuple):
    """
    Counts the number of particles at a distance m from a given index in the lattice.

    :param lattice: Lattice with particles, array
    :param m: Distance to count neighbours at, int
    :param idx: Index of the particle to count neighbours for, tuple (x, y)

    :return: Number of neighbours at distance m, int
    """
    
    #Initialise required parameters for method
    x, y = idx
    Lx, Ly = np.shape(lattice)
    dist_m_neighbours = 0
    offsets = offset_L1(m)

    #Separated dx,dy offsets and get positions to check for occupied particles
    dxs = np.array([dx for dx, dy in offsets])
    dys = np.array([dy for dx, dy in offsets])
    xs = (x + dxs) % Lx
    ys = (y + dys) % Ly

    #Count occupied positions at distance m away from given cell
    dist_m_neighbours = np.count_nonzero(lattice[xs, ys] > 0)

    #Cast a np.int64 to avoid overflow
    dist_m_neighbours = np.int64(dist_m_neighbours)

    return dist_m_neighbours

def count_pair_distances(lattice: np.ndarray, m: int) -> int:
    """
    Counts the number of pairs of particles separated by a distance m (in the L^1 sense).

    :param lattice: Lattice with particles, array
    :param m: Distance to count pairs at, int

    :return: Number of pairs separated by distance m, int
    """

    # Get the indices of all particles in the lattice
    particle_indices = np.argwhere(lattice > 0)

    # Initialise a counter for pairs
    pair_count = 0

    #count pairs using counting function for each individiual particle
    for i in range(len(particle_indices)):
        pair_count += mdist_neighbours(lattice, m, tuple(particle_indices[i]))

    #divide by 2 due to double counting
    return pair_count/2

def PCF(lattice: np.ndarray, m: int) -> float:
    """
    Calculate the pair correlation function (PCF) for a given lattice and distance m.

    :param lattice: The input lattice (2D numpy array)
    :param m: The distance for which to calculate the PCF

    :return: The PCF value for the given distance (float)
    """

    #Get dimensions of lattice
    Lx, Ly = np.shape(lattice)

    #Total number of occupied sites in the lattice
    N = int(np.count_nonzero(lattice))

    #Calculate PCF
    pcf = ( count_pair_distances(lattice, m)  / (2*np.int64(m)*N*(N-1)) ) * (Lx*Ly - 1)

    return pcf