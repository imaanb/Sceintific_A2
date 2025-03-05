import numpy as np
import matplotlib.pyplot as plt
import random
import imageio

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

def random_walk(N, p=0.7, max_iter = 1000000):
    grid = np.zeros((N+1,N+1))
    init = np.random.choice(N+1)
    grid[N, init] = 1

    cluster_growth = []
    walkers = []

    pos_y = 0
    pos_x = np.random.choice(N+1)

    walkers.append((pos_x,pos_y))
    for i in range(1,max_iter):
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
            if random.uniform(0,1) < p: 
                grid[new_pos_y, new_pos_x] = 1
            else:
                new_directions = [d for d in directions if not np.array_equal(d, dir)]
                new_dir = random.choice(new_directions)
                pos_x, pos_y = (new_pos_x + new_dir[0]) % N, new_pos_y + new_dir[1]
                continue

        cluster_growth.append(grid)

        if np.min(grid) == 1:
            return cluster_growth, walkers
        pos_x = new_pos_x
        pos_y = new_pos_y
    return cluster_growth, walkers 


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
                ax.plot(x, rows - y , color='black', marker='o', markersize=100/N)

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

def plot_ps(ps):
    total_grid_sizes = []
    for p in ps:
        grid_sizes = []
        for i in range(10):
            cluster_growth, route = random_walk(50, p)
            final_grid = cluster_growth[-1]
            unique, counts = np.unique(final_grid, return_counts=True)
            grid_sizes.append(counts[1])
        print("Done with", p)
        total_grid_sizes.append(np.mean(grid_sizes))
    print(total_grid_sizes)
    plt.plot(ps,total_grid_sizes)
    plt.xlabel("Sticking probability $p_s$")
    plt.ylabel("Mean clustersize")
    plt.show()

def create_gif_rw(grids, route ,N ,filename='random_walker_5.gif', interval=100):
    """
    Create a GIF of the cluster spreading and the route of the walker over time.

    Parameters:
    grid (dict): Dictionary containing the total grid (NxN) at each timestep.
    route (dict): Dictionary containing the position of the walker at each timestep.
    N (int): Grid size
    filename (str): Name of the output GIF file.
    interval (int): Time interval between frames in milliseconds.
    """
    frames = []
    
    for timestep in sorted(grids.keys()):

        walker_pos = route.get(timestep, None)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grids[timestep], cmap='gray_r', origin='lower')
        
        if walker_pos:
            ax.scatter(walker_pos[1], walker_pos[0], color='red', s=100, label='Walker')
        
        ax.set_title(f'Timestep: {timestep}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.legend()
        plt.tight_layout()
        
        # Save the current frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    
    # Create the GIF
    imageio.mimsave(filename, frames, duration=interval / 1000)



N = 50
ps = [0.5,0.6,0.7,0.8,0.9,1]

cluster_growth, route = random_walk(N, 0.5)
final_grid = cluster_growth[-1]
plot_final_grid(final_grid)