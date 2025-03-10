import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit


@njit
def cluster_in_neigborhood(N, x, y, grid):
    '''
    Checks if the cluster is in the neigborhood of the current location, using the periodic left and right borders.
    '''
    if x==0:
        neighbors = [(y, N-1), (y, x+1), (y-1, x), (y+1, x)]
    elif x==N:
        neighbors = [(y, x-1), (y, 0), (y-1, x), (y+1, x)]
    else:
        neighbors = [(y, x+1), (y, x-1), ( y-1, x), ( y+1, x)]
    for nx, ny in neighbors:
        if 0 <= nx < N and 0 <= ny < N and grid[nx, ny] == 1:
            return True
    return False

@njit
def random_walk(N, p, max_iter = 1000000000):
    '''
    Models the cluster growth with a random walk approach. 
    A random walker is released into a grid and walk untill he is in the neigborhood of the cluster, it then stick to the cluster with probability p.
    '''
    grid = np.zeros((N,N), dtype=np.int8)
    init = (N)//2
    grid[N-1, init] = 1

    pos_y = 0
    pos_x = np.random.randint(0,N)

    directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

    for i in range(1,max_iter):
        dir_index = np.random.randint(0,4)
        dir  = directions[dir_index]

        new_pos_x, new_pos_y = (pos_x + dir[0]) % N, pos_y + dir[1]
        if new_pos_y == N or new_pos_y == -1:
            new_pos_y = 0
            new_pos_x = np.random.randint(0,N)

        
        if cluster_in_neigborhood(N, new_pos_x, new_pos_y, grid):
            if random.uniform(0,1) < p: 
                grid[new_pos_y, new_pos_x] = 1
                if new_pos_y == 0:
                    break
                pos_y, pos_x = 0, np.random.randint(N)
                continue
            else:
                new_directions = [d for d in directions if not np.array_equal(d, dir)]
                new_dir_index = np.random.randint(0,3)
                new_dir = new_directions[new_dir_index]
                pos_x, pos_y = (new_pos_x + new_dir[0]) % N, new_pos_y + new_dir[1]
                continue


        pos_x = new_pos_x
        pos_y = new_pos_y
    return grid.copy() 


def plot_final_grid(grid, N, ax=None):
    '''
    Plots the Cluster in the final grid
    '''
    rows, cols = grid.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.imshow(grid, cmap='Greys', interpolation='nearest')
    ax.axis('off')

    # Formatting
    ax.set_xlim(-1, cols+1)
    ax.set_ylim(-1, rows+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_frame_on(False)

def plot_subplots(N, p_values):
    ''' 
    Combines subplot to one plot
    '''
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows, 2 columns
    axes = axes.flatten()
    
    for i, p in enumerate(p_values):
        final_grid = random_walk(N, p)

        plot_final_grid(final_grid, N, ax=axes[i]) 
        axes[i].set_title(f"$p_s$ = {p}", fontsize = 20)
    
    plt.tight_layout()
    plt.show()

def counter(grid, box_size): 
    ''' 
    
    '''
    grid_size = grid.shape[0]
    count = 0 
    for x in range(0, grid_size, box_size): 
        for y in range(0, grid_size, box_size): 
            box = grid[x: x+box_size, y: y + box_size]
            if np.any(box==1) and np.any(box==0):
                count += 1

    return count

def fractal_dimension(grid): 
    min_size = 2
    max_size = grid.shape[0]
    sizes = []
    counts = [] 
    size = min_size
    while size <= max_size:
        count = counter(grid, size)
        sizes.append(size)
        counts.append(count)
        size *= 2

    sizes = np.array(sizes)
    counts = np.array(counts)

    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    D_box = - coeffs[0]
    return D_box