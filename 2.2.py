import numpy as np
import matplotlib.pyplot as plt
import random

def cluster_in_neigborhood(N, x, y, grid):
    if x==0:
        neighbors = [(y, N), (y, x+1), (y-1, x), (y+1, x)]
    elif x==N:
        neighbors = [(y, x-1), (y, 0), (y-1, x), (y+1, x)]
    else:
        neighbors = [(y, x+1), (y, x-1), ( y-1, x), ( y+1, x)]
    for nx, ny in neighbors:
        if 0 <= nx <= N and 0 <= ny <= N and grid[nx, ny] == 1:
            return True
    return False

def random_walk(N, max_iter = 1000):
    grid = np.zeros((N+1,N+1))
    init = np.random.choice(N+1)
    grid[N, init] = 1

    all_grids = [grid]

    pos_y = 0
    pos_x = np.random.choice(N+1)

    walkers = [(pos_x,pos_y)]
    for i in range(max_iter):
        # dir = np.random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        directions = [np.array([-1, 0]), 
          np.array([0, 1]),
          np.array([1, 0]),
          np.array([0, -1])]
        dir  = random.choice(directions)

        new_pos_x, new_pos_y = (pos_x + dir[0]) % N, pos_y + dir[1]
        if new_pos_y == N + 1 or new_pos_y == -1:
            new_pos_y = 0
            new_pos_x = np.random.choice(N+1)

        walkers.append((new_pos_x,new_pos_y))
        
        if cluster_in_neigborhood(N, new_pos_x, new_pos_y, grid):
            grid[new_pos_y, new_pos_x] = 1

        all_grids.append(grid)

        if np.min(grid) == 1:
            return all_grids, walkers
        pos_x = new_pos_x
        pos_y = new_pos_y
    return all_grids, walkers 


def plot_final_grid(grid):
    rows, cols = grid.shape

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot grid lines
    for x in range(0,cols):
        ax.plot([x, x], [1, rows], 'k-', lw=0.5)  # Vertical lines
    for y in range(1,rows+1):
        ax.plot([0, cols-1], [y, y], 'k-', lw=0.5)  # Horizontal lines

    # Plot filled circles at intersections where grid == 1
    for x in range(cols):
        for y in range(rows):
            if grid[y, x] == 1: 
                ax.plot(x, rows - y , color='black', marker='o', markersize=8)

    # Formatting
    ax.set_xlim(-1, cols+1)
    ax.set_ylim(-1, rows+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_frame_on(False)

    plt.show()

def plot_walker(route):
    x, y = zip(*route) 
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='red', label='Punten') 
    
    # Draw arrows
    for i in range(len(route) - 1):
        plt.arrow(x[i], y[i], x[i+1] - x[i], y[i+1] - y[i], 
                  head_width=0.1, head_length=0.1, fc='black', ec='black', alpha = 0.2)
    
    # plt.grid()
    plt.show()

N = 5
grids, route = random_walk(N)

plot_walker(route)
# rw2 = random_walk(N)

# plot_grid(rw1)
# plot_grid(rw2)