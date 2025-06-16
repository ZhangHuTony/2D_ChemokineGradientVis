# 2D Chemokine Gradient Visualization and T-Cell Movement Simulation

This project visualizes the 2D diffusion of chemokines within a heterogeneous tumor microenvironment and simulates the stochastic migration of T-cells in response to these gradients. It was developed as the final project for **CS 5635: Visualization for Scientific Data** at the University of Utah.

## 🚀 Overview

* **Chemokine Field**: Modeled as a 2D steady-state diffusion equation with degradation, solved analytically using modified Bessel functions for multiple finite-radius cell sources.
* **T-Cell Simulation**: Cells follow a von Mises-distributed movement pattern biased by the local chemokine gradient.
* **Visualization**: Generated `.vtk` files for Line Integral Convolution (LIC), concentration surfaces, cell glyphs, and animated T-cell pathlines, viewable in ParaView.

This project extends the 1D simulation framework at: [1D-ChemokineGradientSim](https://github.com/ZhangHuTony/1D-ChemokineGradientSim)

## 🧪 Scientific Motivation

The Reeve's Lab at the Huntsman Cancer Institute observed non-random CD4 T-cell localization within "hot" and "hotter" tumor regions. This project investigates whether chemokine gradients alone could account for this behavior by comparing simulated T-cell paths against observed data.

## 📁 Directory Structure

```
project_root/
├── chemokine_vis_generator.py         # Chemokine field + glyph VTK generator
├── tcell_movement_generator.py        # T-cell movement simulation + VTK export
├── 2D_Steady_State_Diffusion_Derivation.pdf  # Analytical derivation of diffusion model
├── Final Project Report.docx          # Written report for CS 5635
├── final project vis.pptx             # Slide presentation for final demo
├── full_data/                         # Raw full-size datasets
├── full_output/                       # Simulation outputs using full_data
├── mini_data/                         # Small-scale data for prototyping/debugging
├── mini_output/                       # Output from simulation on mini_data
```

## 📊 Visualization Features

* **Chemokine Field (LIC)**: Gradient-based LIC shows the flow of chemokines.
* **Chemokine Surface**: 3D surface plot (Z = concentration) illustrates spatial distribution.
* **Cell Glyphs**:

  * Red = RFP-labeled ("hotter") tumor cells
  * Green = YFP-labeled ("hot") tumor cells
  * Blue = CD4 T-cells
* **T-cell Movement**:

  * Paths animated with trail history
  * Sensitivity tunable via κ-scaling (gradient bias factor)
  * Random respawn for boundary exits

## 📘 Model Summary

Chemokine distribution `c(x, y)` is governed by:

```
∇2c - λ²c = 0, where λ² = δ / D
```

Each tumor cell acts as a chemokine source with:

* Finite radius `a`
* Constant intracellular concentration `c_cell`
* Contributions calculated via piecewise modified Bessel function `K₀(λr)`

The total field is a superposition of contributions from all cells.

T-cell movement:

* Gradient-following direction `μ = arctan2(gy, gx)`
* Bias strength `κ = c × ||∇c||`
* Angle sampled via von Mises distribution: `θ ~ VM(μ, κ)`

## 🛠 How to Run

1. Ensure all dependencies are installed:

   ```bash
   pip install numpy pandas scipy
   ```

2. Generate chemokine field VTKs:

   ```bash
   python chemokine_vis_generator.py
   ```

3. Simulate and export T-cell paths:

   ```bash
   python tcell_movement_generator.py
   ```

4. Visualize `.vtk` outputs using [ParaView](https://www.paraview.org/).

## 🧠 Lessons Learned

* Chemokine gradients alone are not sufficient to explain T-cell localization in complex tumor environments.
* Parameter tuning (e.g., κ-scaling) heavily influences behavior—high κ leads to trapping, low κ yields randomness.
* Data generation from scratch introduced many more subtleties than expected compared to off-the-shelf datasets.

## 📌 References

* [Steady-state diffusion derivation](2D_Steady_State_Diffusion_Derivation.pdf)
* \[CS 5635 Final Report]\(Final Project Report.docx)
* \[Project Slides]\(final project vis.pptx)
* [Reeve's Lab Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10168251/)
* [T-cell chemokine sensitivity paper](https://www.frontiersin.org/articles/10.3389/fimmu.2022.913366/full)

## 👨‍💻 Author

Tony Zhang | u1183156
University of Utah – Data Science B.S.

---

For questions or collaboration: [GitHub Profile](https://github.com/ZhangHuTony)
