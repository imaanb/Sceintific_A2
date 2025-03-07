import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit


@njit
def cluster_in_neigborhood(N, x, y, grid):
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
    grid = np.zeros((N,N), dtype=np.int8)
    init = (N)//2
    grid[N-1, init] = 1

    # cluster_growth = []
    # walkers = []

    pos_y = 0
    pos_x = np.random.randint(0,N)

    # walkers.append((pos_x,pos_y))

    directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

    for i in range(1,max_iter):
        dir_index = np.random.randint(0,4)
        dir  = directions[dir_index]

        new_pos_x, new_pos_y = (pos_x + dir[0]) % N, pos_y + dir[1]
        if new_pos_y == N or new_pos_y == -1:
            new_pos_y = 0
            new_pos_x = np.random.randint(0,N)

        # walkers.append((new_pos_x,new_pos_y))
        
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

        # cluster_growth.append(grid.copy())

        pos_x = new_pos_x
        pos_y = new_pos_y
    return grid.copy() #cluster_growth #, walkers 


# def plot_final_grid(grid, N, ax = None):
#     rows, cols = grid.shape

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6, 6))

#     # Plot filled circles at intersections where grid == 1
#     for x in range(cols):
#         for y in range(rows):
#             if grid[y, x] == 1: 
#                 ax.plot(x, rows - y , color='black', marker='o', markersize=150/N)

#     # Formatting
#     ax.set_xlim(-1, cols+1)
#     ax.set_ylim(-1, rows+1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.grid(False)
#     ax.set_frame_on(False)

#     # plt.show()

def plot_final_grid(grid, N, ax=None):
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

def plot_six_subplots(N, p_values):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows, 2 columns
    axes = axes.flatten()
    
    for i, p in enumerate(p_values):
        # cluster_growth = random_walk(N, p)  # Generate data
        # final_grid = cluster_growth[-1]  # Extract final grid
        final_grid = random_walk(N, p)

        plot_final_grid(final_grid, N, ax=axes[i])  # Pass ax to function
        axes[i].set_title(f"$p_s$ = {p}", fontsize = 20)
    
    plt.tight_layout()
    plt.show()

def counter(grid, box_size): 
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



N = 100
# plot_final_grid(random_walk(N,0.5),N)
# plt.show()

p_values = np.arange(0.01, 1.01, 0.01)
num_p = len(p_values)
iter = 1
fract = np.zeros(num_p)
j = 0
for p in p_values:
    fract_p = np.zeros(iter)
    for i in range(iter):
        grid = random_walk(N,p)
        fract_p[i] = fractal_dimension(grid)
    fract[j] = np.mean(fract_p)
    j+=1
    print("done with", p)

plt.plot(p_values, fract)
plt.xlabel("$p_s$", fontsize=20)
plt.ylabel("Fractal dimension", fontsize=20)
plt.grid(True)
plt.show()

# p_values = [0.25,0.5,0.75,1]
# plot_six_subplots(N, p_values)


