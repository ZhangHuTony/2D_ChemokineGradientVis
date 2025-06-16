import time
import numpy as np
import pandas as pd
from scipy.special import k0, k1
from scipy.interpolate import RegularGridInterpolator

# ===========================
# 1. Data Ingestion and Parameter Setup
# ===========================
# Load CSV files (ensure they contain at least "x" and "y")
df_rfp = pd.read_csv('real_data/RFP_Pos.csv')
df_yfp = pd.read_csv('real_data/YFP_Pos.csv')
df_cd4 = pd.read_csv('real_data/CD4_Pos.csv')

# df_rfp = pd.read_csv('mini_data/mini_RFP_Pos.csv')
# df_yfp = pd.read_csv('mini_data/mini_YFP_Pos.csv')
# df_cd4 = pd.read_csv('mini_data/mini_CD4_Pos.csv')

# Assign different cell-specific intracellular concentrations for RFP and YFP.
# CD4 cells do not contribute to chemokine production.
df_rfp['c_cell'] = 4   # RFP gets higher concentration
df_yfp['c_cell'] = 1    # YFP gets lower concentration

# Define cell radii (in same units as grid)
df_rfp['radius'] = 0.01
df_yfp['radius'] = 0.01
# CD4 cells need not have a chemokine value, but we define a radius for position purposes (if desired)
df_cd4['radius'] = 0.01

# Set cell type codes: 0 for RFP, 1 for YFP, and 2 for CD4
df_rfp['cell_type'] = 0  
df_yfp['cell_type'] = 1  
df_cd4['cell_type'] = 2

# Combine the cancer cells (RFP and YFP) for the chemokine simulation
df_cells = pd.concat([df_rfp, df_yfp], ignore_index=True)

# Combine all cell types for glyph export (cancer cells + CD4)
df_all_cells = pd.concat([df_rfp, df_yfp, df_cd4], ignore_index=True)

# Diffusion and degradation parameters for the chemokine.
D = 128              # Diffusion coefficient
delta = 0.00542           # Degradation rate
lam = np.sqrt(delta/D)  # λ = sqrt(δ/D)

# ===========================
# 2. Create a Spatial Grid for the Chemokine Field
# ===========================
# Define grid boundaries with some margin around the cancer cells (RFP and YFP).
margin = 10
xmin = df_cells['x'].min() - margin
xmax = df_cells['x'].max() + margin
ymin = df_cells['y'].min() - margin
ymax = df_cells['y'].max() + margin

grid_spacing = 10.0  # adjust according to desired resolution
nx = int(np.ceil((xmax - xmin) / grid_spacing)) + 1
ny = int(np.ceil((ymax - ymin) / grid_spacing)) + 1


x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)  # Shapes (ny, nx)

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# Initialize fields for the chemokine concentration and its gradient.
C_total = np.zeros_like(X)
gradX_total = np.zeros_like(X)
gradY_total = np.zeros_like(X)

# ===========================
# 3. Compute the Chemokine Field and Its Gradient (Timed)
# ===========================
start_time = time.perf_counter()

# Loop over each cancer cell (RFP and YFP) and add its contribution.
for _, cell in df_cells.iterrows():
    x0 = cell['x']
    y0 = cell['y']
    c_cell = cell['c_cell']
    a = cell['radius']
    
    # Compute Euclidean distance from the cell center to each grid point.
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
    # Temporary arrays for this cell's contribution.
    C = np.zeros_like(X)
    gradX = np.zeros_like(X)
    gradY = np.zeros_like(X)
    
    # Define inside/outside masks.
    inside = R < a
    outside = R >= a
    
    # --- Concentration Field ---
    # Inside the cell: constant concentration.
    C[inside] = c_cell
    # Outside the cell: use the modified Bessel function piecewise formula.
    k0_la = k0(lam * a)
    C[outside] = c_cell * k0(lam * R[outside]) / k0_la
    
    # --- Gradient Computation (only for outside points) ---
    if np.any(outside):
        R_out = R[outside]
        factor = -c_cell * lam / k0_la * k1(lam * R_out)
        gradX[outside] = factor * ((X[outside] - x0) / R_out)
        gradY[outside] = factor * ((Y[outside] - y0) / R_out)
    
    # Add this cell's contributions to the overall field.
    C_total += C
    gradX_total += gradX
    gradY_total += gradY

end_time = time.perf_counter()
print("Chemokine field calculation took {:.4f} seconds".format(end_time - start_time))

# Create an interpolator for the chemokine field.
interp_func = RegularGridInterpolator((y, x), C_total)

# ===========================
# 4. Write VTK Files for the Chemokine Field
# ===========================
# 4A. Write the flat vector field (for LIC visualization).
npoints = nx * ny
vtk_filename = "chemokine.vtk"
with open(vtk_filename, "w") as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Chemokine Concentration and Gradient Data\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write("DIMENSIONS {} {} 1\n".format(nx, ny))
    f.write("ORIGIN {} {} 0\n".format(xmin, ymin))
    f.write("SPACING {} {} 1\n".format(dx, dy))
    f.write("POINT_DATA {}\n".format(npoints))
    f.write("SCALARS concentration float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for value in C_total.ravel(order='C'):
        f.write("{}\n".format(value))
    f.write("VECTORS gradient float\n")
    flat_gradX = gradX_total.ravel(order='C')
    flat_gradY = gradY_total.ravel(order='C')
    for gx, gy in zip(flat_gradX, flat_gradY):
        f.write("{} {} {}\n".format(gx, gy, 0.0))
print("Flat vector field VTK file saved as '{}'".format(vtk_filename))

# 4B. Write the 3D surface version (Structured Grid) where z = concentration.
vtk_surface_filename = "chemokine_surface.vtk"
with open(vtk_surface_filename, "w") as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Chemokine Concentration Surface (Height = concentration)\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_GRID\n")
    f.write("DIMENSIONS {} {} 1\n".format(nx, ny))
    npoints = nx * ny
    f.write("POINTS {} float\n".format(npoints))
    for j in range(ny):
        for i in range(nx):
            f.write("{} {} {}\n".format(X[j, i], Y[j, i], C_total[j, i]))
    f.write("\nPOINT_DATA {}\n".format(npoints))
    f.write("SCALARS concentration float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for value in C_total.ravel(order='C'):
        f.write("{}\n".format(value))
print("3D surface VTK file saved as '{}'".format(vtk_surface_filename))

# ===========================
# 5. Write VTK Files for Cell (Glyph) Positions
# Two versions will be generated:
# - cells_interpolated.vtk: All cell types (RFP, YFP, and CD4/T cells) have their z coordinate sampled from the chemokine field.
# - cells_flat.vtk: All cell glyphs are placed at z = 0.
# In both files, a scalar field "cell_type" is added for subsequent color-coding.
# ===========================

def write_cells_vtk(filename, df, use_interpolated_z=True):
    """
    Writes a VTK polydata file for cell positions.
    If use_interpolated_z is True, the z coordinate is obtained by interpolating the chemokine field
    at the (x, y) position for all cells.
    Otherwise, all cells are given z = 0.
    """
    num_cells = len(df)
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Cell Positions with cell_type attribute\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS {} float\n".format(num_cells))
        for _, cell in df.iterrows():
            x0, y0 = cell['x'], cell['y']
            if use_interpolated_z:
                # For all cell types, use the interpolated chemokine field value.
                z_val = float(interp_func((y0, x0)))
            else:
                z_val = 0.0
            f.write("{} {} {}\n".format(x0, y0, z_val))
        f.write("\nVERTICES {} {}\n".format(num_cells, num_cells * 2))
        for i in range(num_cells):
            f.write("1 {}\n".format(i))
        # Write a scalar field to encode cell type for coloring.
        f.write("\nPOINT_DATA {}\n".format(num_cells))
        f.write("SCALARS cell_type int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _, cell in df.iterrows():
            f.write("{}\n".format(int(cell['cell_type'])))

# Write the version with interpolated z for all cells (RFP, YFP, CD4/T cells).
write_cells_vtk("cells_interpolated.vtk", df_all_cells, use_interpolated_z=True)
print("Cells (interpolated height) VTK file saved as 'cells_interpolated.vtk'.")

# Write the version with z = 0 for all cells.
write_cells_vtk("cells_flat.vtk", df_all_cells, use_interpolated_z=False)
print("Cells (flat, z=0) VTK file saved as 'cells_flat.vtk'.")
