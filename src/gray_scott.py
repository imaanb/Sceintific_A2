import numpy as np
import matplotlib.pyplot as plt

def initialize_system(nx, ny, noise_amplitude=0.01):
    """Initialize the system and add a small amount of noise."""

    # Initialize u with 0.5 everywhere and v with 0
    u = np.ones((nx, ny)) * 0.5
    v = np.zeros((nx, ny))
    
    # Create a small square of v=0.25 in the center
    square_size = min(nx, ny) // 10
    x_center, y_center = nx // 2, ny // 2
    x_start, x_end = x_center - square_size // 2, x_center + square_size // 2
    y_start, y_end = y_center - square_size // 2, y_center + square_size // 2
    
    v[x_start:x_end, y_start:y_end] = 0.25
    
    # Add small random noise to break symmetry
    u += noise_amplitude * (np.random.random((nx, ny)) - 0.5)
    v += noise_amplitude * (np.random.random((nx, ny)) - 0.5)
    
    return u, v

def laplacian(Z, dx=1.0):
    """Laplacian using a 5-point stencil + periodic boundary conditions."""
    Ztop = np.roll(Z, -1, axis=0)
    Zbottom = np.roll(Z, 1, axis=0)
    Zleft = np.roll(Z, 1, axis=1)
    Zright = np.roll(Z, -1, axis=1)
    
    return (Ztop + Zbottom + Zleft + Zright - 4 * Z) / (dx * dx)

def gray_scott_update(u, v, Du, Dv, f, k, dt=1.0, dx=1.0):
    """
    Update the Gray-Scott model by one timestep.
    """
    #Compute Laplacians and Reaction terms
    laplacian_u = laplacian(u, dx)
    laplacian_v = laplacian(v, dx)
    reaction_u = -u * v * v + f * (1 - u)
    reaction_v = u * v * v - (f + k) * v
    
    #Updates
    u_new = u + dt * (Du * laplacian_u + reaction_u)
    v_new = v + dt * (Dv * laplacian_v + reaction_v)
    
    #ensure non-negative
    u_new = np.maximum(u_new, 0)
    v_new = np.maximum(v_new, 0)
    
    return u_new, v_new

def simulate_gray_scott(nx=200, ny=200, steps=5000, Du=0.16, Dv=0.08, f=0.035, k=0.060, dt=1.0, dx=1.0):
    """Simulate the Gray-Scott model and return final state."""
    
    u, v = initialize_system(nx, ny)
    
    for step in range(steps):
        u, v = gray_scott_update(u, v, Du, Dv, f, k, dt, dx)
        
    return u, v

def main():
    """Main function to run the Gray-Scott model simulations and display all results in one figure."""
    
    Du = 0.16
    Dv = 0.08
    dt = 1.0
    dx = 1.0
    f_list = [0.022, 0.026, 0.035]
    k_list = [0.051, 0.053, 0.060]
    
    plt.figure(figsize=(15, 10))
    
    for idx, f in enumerate(f_list):
        k = k_list[idx]
        
        # Run simulation
        u, v = simulate_gray_scott(Du=Du, Dv=Dv, f=f, k=k, dt=dt, dx=dx)
        
        #plot U
        plt.subplot(3, 2, 2*idx + 1)
        im = plt.imshow(u, cmap='viridis')
        plt.colorbar(im, label='Concentration of U')
        plt.title(f'U Concentration (f={f}, k={k})')
        
        #plot V
        plt.subplot(3, 2, 2*idx + 2)
        im = plt.imshow(v, cmap='viridis')
        plt.colorbar(im, label='Concentration of V')
        plt.title(f'V Concentration (f={f}, k={k})')
    
    plt.tight_layout()
    plt.savefig(f"results/combined_gray_scott_patterns")
    plt.show()

if __name__ == "__main__":
    main()