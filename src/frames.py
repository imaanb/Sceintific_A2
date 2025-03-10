import numpy as np
import random as rd
import matplotlib.pyplot as plt
import imageio
from scipy.special import erfc
import os
import pickle



# ## ANALYTIC SOLUTION FOR COMPARISON
def analytic_solution(x, t, D, num_terms):
    if t == 0:
        return 0  # Boundary condition
    solution = 0
    for i in range(num_terms):
        term1 = erfc((1 - x + 2 * i) / (2 * np.sqrt(D * t)))
        term2 = erfc((1 + x + 2 * i) / (2 * np.sqrt(D * t)))
        solution += term1 - term2
    return solution

def SOR_iteration(c, w, N, epsilon = 1e-5, max_iter=10000):
    c_next = c.copy()
    results = {}

    for k in range(max_iter): 
        for j in range(1, N):  # Skip boundaries in x-direction
            for i in range(1, N):  # Skip boundaries in y-direction
                c_next[i, j] = (
                    w * 0.25
                    * (c[i + 1, j] + c_next[i - 1, j] + c[i, j + 1] + c_next[i, j - 1])
                    + (1 - w) * c[i,j]
                )

        # Apply periodic boundary conditions in the x-direction
        for j in range (1, N):
            c_next[N,j] = (
                    w * 0.25
                    * (c[1, j] + c_next[N - 1, j] + c[N, j + 1] + c_next[N, j - 1])
                    + (1 - w) * c[N, j]
                )
            
        c_next[0, 1:N] = c_next[N, 1:N]

        # Apply fixed boundary conditions
        c_next[:, N] = 1.0  # Top boundary
        c_next[:, 0] = 0.0  # Bottom boundary

        delta = np.max(np.abs(c_next-c))

        results[k] = {"c": c_next.copy()[0], "delta": delta}

        if delta < epsilon:
            #print("SOR: ", k)
            return c_next, results
        else:
            c[:] = c_next[:]

def get_neighbours(i,j, N): 
    neighbours = [[i, j-1],[i, j+1],[i+1, j],[i-1, j]] 
    return [[ni, nj] for ni, nj in neighbours if 0 <= ni <= N and 0 <= nj <= N]

def get_growth_candidates(grid, N): 
    candidates = set()
    for i in range(N+1): 
        for j in range(N+1): 
            if grid[i, j]:
                for ni, nj in get_neighbours(i, j, N):
                    if not grid[ni, nj]:
                        candidates.add((ni, nj))   
    
    return list(candidates)
    

def get_p(diffusion_grid, candidates, eta): 
    weights = np.array([diffusion_grid[i, j]**eta for i, j in candidates])
    return weights / np.sum(weights)  # Normalize probabilities

def dla(c, cluster_grid, w, N, timesteps , eta, epsilon = 1e-5, max_iter=10000): 
    growth_evolution = {}
    timestep = 0
    c_j = 0 
    while c_j < N:         
        diffusion_grid, _ = SOR_iteration(c, w, N, epsilon, max_iter)
        
        candidates = get_growth_candidates(cluster_grid, N)

        if not candidates:
            print(f"No grotwh possibilities for the cluster left (after {timestep} timesteps)")
            break 
        p = get_p(diffusion_grid, candidates, eta)
        
        c_i, c_j = candidates[np.random.choice(len(candidates), p = p)]
        cluster_grid[c_i, c_j] = 1 
        
        growth_evolution[timestep] = np.rot90(cluster_grid.copy(), k=-3)
        
        c = diffusion_grid.copy()
        timestep += 1

    return growth_evolution, timestep
        

def counter(grid, box_size): 
    grid_size = grid.shape[0]
    count = 0 
    for x in range(0, grid_size, box_size): 
        for y in range(0, grid_size, box_size): 
            box = grid[x: x+box_size, y: y + box_size]
            if np.any(box == 1) and np.any(box == 0):
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

show_etas = [.01, 1, 10, 30]

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

def plot_six_subplots(N, etas, steps):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows, 2 columns
    axes = axes.flatten()
    
    for i, p in enumerate(etas):
        ## initialize ## 
        # Calculate the analytic solution
        c = np.zeros((N+1, N+1))
        c[:, N] = 1.0  # Top boundary fixed to 1.0
        c[:, 0] = 0.0  # Bottom boundary fixed to 0.0
        c[:] = np.linspace(0, 1, N+1).reshape(-1, 1)  # Analytical linear concentration gradient: c(y) = y for t → ∞

        cluster_grid = np.zeros((N+1, N+1))  # diffusion grid 
        cluster_grid[N//2, 0] = 1 
        
        growth_evolution, k = dla(c, cluster_grid, 1.75, N, steps, p)
        max_timestep = max(growth_evolution.keys())
        final_frame = growth_evolution[max_timestep]
        final_grid = np.flipud(final_frame)

        fractal_dim= fractal_dimension(final_grid)
        plot_final_grid(final_grid, N, ax=axes[i])  # Pass ax to function
        axes[i].set_title(f"eta = {p}, fractal dimension = {fractal_dim:.2f}", fontsize=16)
    plt.suptitle("Final frames for different eta", fontsize = 20)
    plt.tight_layout()
    plt.savefig("results/final_frames_for_different_eta_500._earlystop.png", dpi = 300)
    plt.show()

def main():
    plot_six_subplots(100, show_etas, steps=500)

    etas = np.arange(0, 30, .25 )
    """
    under1 = np.arange(0, 1, .01)
    over1 = np.arange(1, 30, .25)
    etas = np.concatenate((under1, over1))
    """
    etas = etas[1:]

    def plotter(filename, etas):
        with open(os.path.join("final_runs", filename), 'rb') as f:
            fractal_dims = pickle.load(f)

        fracdims = [fractal_dims[eta] for eta in etas]
        print(f"maximum fractal dimension: {max(fracdims)} at {etas[np.argmax(fracdims)]}")

        plt.figure(figsize=(15,10))
        plt.plot(etas, fracdims)


        plt.xlabel('Eta', fontsize = 16 )
        plt.ylabel('Fractal Dimension', fontsize = 16 )
        plt.title('DLA growth model: Fractal Dimension (t = 500) vs Eta parameter on a 100x100 grid ', fontsize = 20)
        plt.grid(True)
        plt.savefig("runs/dla_fracdim_earlystop", dpi = 300 )
        plt.show()


    filename = "final_early_stop.pkl"

    plotter(filename, etas)

if __name__ == "__main__":
    main()