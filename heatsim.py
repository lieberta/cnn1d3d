import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01  # Thermal diffusivity
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Dimensions of the room
Nx, Ny, Nz = 20, 20, 20  # Number of spatial grid points in each dimension
T = 1.0  # Total simulation time
Nt = 100  # Number of time steps
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)
dt = T / Nt

# Initialize temperature array
u = np.zeros((Nt + 1, Nx, Ny, Nz))

# Set initial condition (e.g., a hot spot in the center)
u[0, :, :, :] = 26.0
u[:, 9:10, 9:10, 0] = 1000.0

# Finite difference scheme
for n in range(Nt):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                laplacian = (
                    (u[n, i + 1, j, k] - 2 * u[n, i, j, k] + u[n, i - 1, j, k]) / dx**2 +
                    (u[n, i, j + 1, k] - 2 * u[n, i, j, k] + u[n, i, j - 1, k]) / dy**2 +
                    (u[n, i, j, k + 1] - 2 * u[n, i, j, k] + u[n, i, j, k - 1]) / dz**2
                )
                u[n + 1, i, j, k] = u[n, i, j, k] + alpha * dt * laplacian

# Export the data (optional, you can use it for further analysis or visualization)
np.savez('heat_equation_solution.npz', time=np.linspace(0, T, Nt + 1), x=np.linspace(0, Lx, Nx),
         y=np.linspace(0, Ly, Ny), z=np.linspace(0, Lz, Nz), temperature=u)

# Plot the results
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y, Z = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny), np.linspace(0, Lz, Nz), indexing='ij')

#ax.plot_surface(X[:, :, Nz//2], Y[:, :, Nz//2], u[-1, :, :, Nz//2], cmap='viridis')

it=99 # Plotted Timestep
iz=0 # Plotted Z-coordinate
ax.plot_surface(X[:, :, iz], Y[:, :, iz], u[it, :, :, iz], cmap='viridis')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature')
ax.set_title(f'Temperature Distribution in 3D Room at t={it} and z={iz}')
plt.show()