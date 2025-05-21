# Attraction Between Like-Poles of Magnets Mediated by a Ferromagnetic Sphere

This repository contains a numerical simulation of the counterintuitive physical phenomenon where two magnets with like poles can attract each other when a ferromagnetic sphere is placed between them. The code implements the theoretical model described in the paper "Attraction between like-poles of two magnets mediated by a soft ferromagnetic material" by Teles et al. (2024).

## Contents

- **simulacion_atraccion_magnetica.py**: Core implementation of the magnetic force simulation
- **animacion_fuerza.py**: Interactive animation of the force vs. distance relationship
- **comparacion_experimental.py**: Comparison between theoretical predictions and experimental data

## Physical Phenomenon

In conventional magnetic interactions, like poles repel each other. However, when a soft ferromagnetic sphere is placed between two magnets with like poles facing each other, a fascinating effect occurs:

- At large distances: The system behaves conventionally, with magnets repelling each other
- At short distances: The magnets unexpectedly attract each other

This phenomenon is due to the redistribution of the magnetic field by the ferromagnetic sphere, which can be modeled using the magnetic image charge method.

## Features

### 1. Core Simulation (`simulacion_atraccion_magnetica.py`)

- Implements the magnetic image charge method for a sphere with high magnetic permeability
- Calculates the force between magnets based on Equation 18 from the reference paper
- Generates various visualizations:
  - Force vs. distance curves
  - 3D visualization of charge and image charge configuration
  - Magnetic force field visualization
  - Parametric analysis with different sphere radii

### 2. Interactive Animation (`animacion_fuerza.py`)

- Provides a dynamic visualization of the changing force as magnets approach each other
- Dual-panel display showing:
  - Quantitative force-distance graph with current position indicator
  - Schematic representation of the physical system with force arrows
- Color-coding (blue for repulsion, red for attraction) to visually indicate force direction

### 3. Experimental Comparison (`comparacion_experimental.py`)

- Compares theoretical predictions with simulated experimental data
- Reproduces Figures 7 and 8 from the reference paper
- Demonstrates how sphere radius affects force magnitude and transition point

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/magnetic-attraction-simulation.git
cd magnetic-attraction-simulation
```

2. Install the required dependencies:
```bash
pip install numpy matplotlib scipy
```

## Usage

### Basic Simulation
```bash
python simulacion_atraccion_magnetica.py
```
This will generate force vs. distance plots, a 3D visualization of the system, and a magnetic force field plot.

### Interactive Animation
```bash
python animacion_fuerza.py
```
This will display an interactive animation showing how the force changes as the distance between the magnet and the sphere varies.

### Experimental Comparison
```bash
python comparacion_experimental.py
```
This will generate plots comparing the theoretical model with simulated experimental data for different sphere radii.

## Customization

You can modify various physical parameters in each script to explore different configurations:

- Sphere radius (`a`)
- Magnet radius (`rm`)
- Magnet length (`dL`, `dR`)
- Magnetic field strength (`B`)
- Distance ranges

## Results

The simulation demonstrates key physical insights:

1. **Bimodal Force Behavior**: At short distances, the magnets experience a strong attractive force; beyond a critical point, they transition to repulsion.

2. **Sphere Size Effect**: Larger spheres intensify both the maximum attractive force and the sharpness of the transition to the repulsive regime.

3. **Force Field Visualization**: The code generates vector field visualizations showing how the ferromagnetic sphere distorts the magnetic field lines.

4. **3D Charge Configuration**: The visualization of real and image charges helps understand the underlying physical mechanism.

## Citation

If you use this code in your research, please cite the original paper:

```
Teles, T. N., de Araújo, T. V. P., & Levin, Y. (2024). Attraction between like-poles of two magnets mediated by a soft ferromagnetic material. Revista Brasileira de Ensino de Física, 46, e20240255.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This simulation is based on the theoretical model developed by Teles, de Araújo, and Levin (2024)
- The code architecture is designed for educational purposes, to illustrate electromagnetic principles in an intuitive way
