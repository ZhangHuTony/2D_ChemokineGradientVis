import time
import numpy as np
import pandas as pd
import os
from scipy.special import k0, k1
from scipy.interpolate import RegularGridInterpolator

# Set the seed for reproducibility.
np.random.seed(42)

# =====================================================
# 1. Helper Functions
# =====================================================

def sample_von_mises_direction(gx, gy, c):
    """
    Samples a new movement angle based on the local gradient using the von Mises distribution.
    
    Parameters:
      gx, gy : float
               The components of the local gradient.
      c      : float
               Scaling constant for determining the base kappa from the gradient magnitude.
    
    Returns:
      theta  : float
               The new movement angle in radians.
    """
    mu = np.arctan2(gy, gx)
    grad_mag = np.sqrt(gx**2 + gy**2)
    kappa = c * grad_mag
    if kappa < 1e-6:
        theta = np.random.uniform(-np.pi, np.pi)
    else:
        theta = np.random.vonmises(mu, kappa)
    return theta

def write_tcells_vtk(filename, positions, interp_conc):
    """
    Writes a VTK polydata file for T-cell positions.
    Each point's z coordinate is determined by sampling the chemokine concentration.
    
    Parameters:
      filename    : output VTK filename.
      positions   : (N,2) array of T-cell x,y coordinates.
      interp_conc : interpolator function for chemokine concentration.
    """
    num_points = positions.shape[0]
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("T-cell positions\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS {} float\n".format(num_points))
        for pos in positions:
            z_val = float(interp_conc((pos[1], pos[0])))
            f.write("{} {} {}\n".format(pos[0], pos[1], z_val))
        f.write("\nVERTICES {} {}\n".format(num_points, num_points * 2))
        for i in range(num_points):
            f.write("1 {}\n".format(i))

def write_tcell_paths_vtk(filename, positions_over_time, t, trail_length, interp_conc, birth_times):
    """
    Writes a VTK polydata file containing polyline paths for T-cells.
    For each T-cell, the path connects its positions from time step max(birth_time, t - trail_length + 1) up to t.
    
    Parameters:
      filename            : output VTK filename.
      positions_over_time : list of (N,2) arrays (each array is positions at a given time step).
      t                   : current time step index.
      trail_length        : number of time steps to include in the path.
      interp_conc         : interpolator for chemokine concentration.
      birth_times         : (N,) array of birth time for each cell.
    """
    n_tcells = positions_over_time[0].shape[0]
    all_points = []  # List to store all 3D points.
    lines = []       # List to store connectivity for each cell.
    
    for i in range(n_tcells):
        polyline_indices = []
        start_index = max(birth_times[i], t - trail_length + 1)
        for t_step in range(start_index, t+1):
            pos = positions_over_time[t_step][i]
            z_val = float(interp_conc((pos[1], pos[0])))
            all_points.append([pos[0], pos[1], z_val])
            polyline_indices.append(len(all_points)-1)
        lines.append(polyline_indices)
    
    num_points = len(all_points)
    num_lines = len(lines)
    total_line_entries = sum(len(line) + 1 for line in lines)
    
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("T-cell paths at timestep {} with trail length {}\n".format(t, trail_length))
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS {} float\n".format(num_points))
        for pt in all_points:
            f.write("{} {} {}\n".format(pt[0], pt[1], pt[2]))
        f.write("\nLINES {} {}\n".format(num_lines, total_line_entries))
        for line in lines:
            f.write("{} {}\n".format(len(line), " ".join(map(str, line))))

# =====================================================
# 2. Data Ingestion and Chemokine Field Setup
# =====================================================
# df_rfp = pd.read_csv('mini_data/mini_RFP_Pos.csv')
# df_yfp = pd.read_csv('mini_data/mini_YFP_Pos.csv')
# df_cd4 = pd.read_csv('mini_data/mini_CD4_Pos.csv')

df_rfp = pd.read_csv('real_data/RFP_Pos.csv')
df_yfp = pd.read_csv('real_data/YFP_Pos.csv')
df_cd4 = pd.read_csv('real_data/CD4_Pos.csv')

df_rfp['c_cell'] = 4    # RFP higher concentration ("RFP heat")
df_yfp['c_cell'] = 1    # YFP lower concentration
df_rfp['radius'] = 0.01
df_yfp['radius'] = 0.01

df_cd4['c_cell'] = 0
df_cd4['radius'] = 0.01

df_rfp['cell_type'] = 0
df_yfp['cell_type'] = 1
df_cd4['cell_type'] = 2

df_cells = pd.concat([df_rfp, df_yfp], ignore_index=True)

# =====================================================
# 3. Create the Regular Spatial Grid and Compute the Chemokine Field
# =====================================================
margin = 10
xmin = df_cells['x'].min() - margin
xmax = df_cells['x'].max() + margin
ymin = df_cells['y'].min() - margin
ymax = df_cells['y'].max() + margin

grid_spacing = 10.0
nx = int(np.ceil((xmax - xmin) / grid_spacing)) + 1
ny = int(np.ceil((ymax - ymin) / grid_spacing)) + 1

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

C_total = np.zeros_like(X)
gradX_total = np.zeros_like(X)
gradY_total = np.zeros_like(X)

D = 128
delta = 0.00542
lam = np.sqrt(delta / D)

start_time = time.perf_counter()
for _, cell in df_cells.iterrows():
    x0 = cell['x']
    y0 = cell['y']
    c_cell = cell['c_cell']
    a = cell['radius']
    
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    C = np.zeros_like(X)
    gradX = np.zeros_like(X)
    gradY = np.zeros_like(X)
    
    inside = R < a
    outside = R >= a
    
    C[inside] = c_cell
    k0_la = k0(lam * a)
    C[outside] = c_cell * k0(lam * R[outside]) / k0_la
    
    if np.any(outside):
        R_out = R[outside]
        factor = -c_cell * lam / k0_la * k1(lam * R_out)
        gradX[outside] = factor * ((X[outside] - x0) / R_out)
        gradY[outside] = factor * ((Y[outside] - y0) / R_out)
    
    C_total += C
    gradX_total += gradX
    gradY_total += gradY

end_time = time.perf_counter()
print("Chemokine field calculation took {:.4f} seconds".format(end_time - start_time))

# Create interpolators with safe bounds handling.
conc_interp = RegularGridInterpolator((y, x), C_total, bounds_error=False, fill_value=0)
gradX_interp = RegularGridInterpolator((y, x), gradX_total, bounds_error=False, fill_value=0)
gradY_interp = RegularGridInterpolator((y, x), gradY_total, bounds_error=False, fill_value=0)

# =====================================================
# 4. Simulate T-Cell Movement with Out-of-Bounds Handling and Respawning
# =====================================================
# Randomly uniformly sample T-cell positions within [xmin, xmax] x [ymin, ymax].
N_tcells = len(df_cd4)
tcell_positions = np.column_stack((
    np.random.uniform(xmin, xmax, N_tcells),
    np.random.uniform(ymin, ymax, N_tcells)
))
n_tcells = tcell_positions.shape[0]

# Array to store the "birth time" for each cell.
# (We keep the original birth time to get a continuous path.)
cell_birth_times = np.zeros(n_tcells, dtype=int)

n_timesteps = 500
speed = 3.0
c_scaling = 10
trail_length = 50

# Instead of directly saving simulation positions,
# we will record the candidate positions (even if out-of-bound)
# to ensure that the movement leaving the domain is captured.
tcell_positions_over_time = [tcell_positions.copy()]

for t in range(n_timesteps):
    # candidate_positions will hold the computed (raw) candidate moves for visualization.
    candidate_positions = np.empty_like(tcell_positions)
    # new_positions will update the simulation state (always in-bound)
    new_positions = tcell_positions.copy()
    
    for i, pos in enumerate(tcell_positions):
        # Compute the candidate new position using the gradient update.
        gx = float(gradX_interp((pos[1], pos[0])))
        gy = float(gradY_interp((pos[1], pos[0])))
        theta = sample_von_mises_direction(gx, gy, c_scaling)
        dx_move = speed * np.cos(theta)
        dy_move = speed * np.sin(theta)
        candidate = pos + np.array([dx_move, dy_move])
        # Always record the candidate position (even if out-of-bound).
        candidate_positions[i] = candidate
        
        # Now update simulation state.
        if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
            # Candidate is out-of-bound.
            # Respawn the cell within the valid domain.
            new_positions[i] = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax)
            ])
            # Do NOT update the birth time so the candidate move is preserved
            # in the path history.
        else:
            new_positions[i] = candidate
            
    # Record the candidate positions for the polyline path even if some are out-of-bound.
    tcell_positions_over_time.append(candidate_positions.copy())
    # Update simulation state with new (in-bound) positions.
    tcell_positions = new_positions.copy()


# =====================================================
# 5. Create Output Folders for Positions and Paths
# =====================================================
rfp_heat = df_rfp['c_cell'].iloc[0]
folder_positions = f"TCells_Positions_D{D}_delta{delta}_RFP{rfp_heat}_cscaling{c_scaling}_speed{speed}"
folder_paths = f"TCells_Paths_D{D}_delta{delta}_RFP{rfp_heat}_cscaling{c_scaling}_speed{speed}"

os.makedirs(folder_positions, exist_ok=True)
os.makedirs(folder_paths, exist_ok=True)

# =====================================================
# 6. Export T-Cell Positions and Path Lines for Visualization
# =====================================================
for t, pos in enumerate(tcell_positions_over_time):
    pos_filename = os.path.join(folder_positions, "Tcells_timestep_{:03d}.vtk".format(t))
    write_tcells_vtk(pos_filename, pos, conc_interp)
    print("Wrote T-cell positions to:", pos_filename)
    
    path_filename = os.path.join(folder_paths, "Tcells_paths_timestep_{:03d}.vtk".format(t))
    write_tcell_paths_vtk(path_filename, tcell_positions_over_time, t, trail_length, conc_interp, cell_birth_times)
    print("Wrote T-cell paths (trail) to:", path_filename)
