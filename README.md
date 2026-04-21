# Ising Model
Python script for simulating the Ising Model on a square (n x n) lattice using Model Carlo methods. Both Glauber (spin-flip) and Kawasaki (spin-exchange) dynamics are supported, with the Metropolis acceptance criterion. 

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Numba

Install dependencies with:

```bash
pip install numpy matplotlib numba
```

## Arguments

The script is run from the terminal using command-line arguments (flags). All arguments are optional — defaults are shown below.

- 'n', Lattice side length. The simulation grid is (n x n), Default = 50
- 'T', Thermal energy k_B T (animation mode only). Default = 2.0
- 'J', Coupling constant J. Default = 1
- 'd', Dynamics: 'g' = Glauber, 'k' = Kawasaki. Default = 'g'
- 'o', Initial state: 'o' = ordered (all up), 'd' = disordered (random). Default = 'd'
- 'm', Run mode: 'a' = animation, 'm' = measurements. Default = 'a'

## Command line examples

### Animation (`--mode ani`)

Runs a live animation of the lattice evolving under the chosen dynamics. The simulation runs for N² total steps (where N = n²), updating the display every 10 sweeps.

```bash
# Default: 50x50 lattice, Glauber dynamics, kbT=2.0, disordered start
python ising_model.py

# Kawasaki dynamics, ordered start, kbT=1.5
python ising_model.py -d k -o o -T 1.5

# Larger lattice, low temperature
python ising_model.py -n 100 -T 1.0
```

### Measurements (`--mode mea`)

Sweeps k_B T from 3.0 down to 1.0 in steps of 0.1. At each temperature, 100 sweeps are used for equilibration, then measurements are taken every 10 sweeps for 10,000 sweeps (1,000 measurements per temperature). Error bars on the specific heat are computed using the Bootstrap method.

```bash
# Glauber measurements, 50x50 lattice, disordered start
python ising_model.py -m m -d g

# Kawasaki measurements, ordered start
python ising_model.py -m m -d k -o o
```

## Output
All outputs are saved relative to the scipt's directory:

```
outputs/
├── datafiles/
│   ├── glauber_energy_and_specific_heat_final.txt
│   ├── glauber_magnetisation_and_susceptibility_final.txt
│   └── kawasaki_energy_and_specific_heat_final.txt
└── plots/
    ├── glauber_energy_and_specific_heat_final_plot.png
    ├── glauber_magnetisation_and_susceptibility_final_plot.png
    └── kawasaki_energy_and_specific_heat_final_plot.png
```
