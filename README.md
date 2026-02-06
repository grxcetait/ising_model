# Ising Model
Python script for simulating the Ising Model in 2D according to Glauber or Kawasaki dynamics.

## ising_model.py
This script can either run an animation or take measurements of the simulation of the Ising Model. Upon running the script, there will be prompts for user input to choose and customise the animation or measurement conditions.

## Instructions for running the code
1. Run the file ising_model.py
    For example: python3 ising_model.py
2. Select between animation ('a') or measurements ('m')
3. Select the system size N (recommended N = 50)
4. In the case of animation, input the temperature T
5. In the case of animation, input the number of iterations to run for (recommended 10000)
6. Select the ordering of the initial lattice state to be ordered ('o') (with all spins pointing up) or disordered ('d') (with all spins randomised)
7. Select the dynamics to be used between Glauber ('g') or Kawasaki ('k')
8. In the case of animation, the animation will run and remain open at the end until the user closes it.
9. In the case of measurements, the output datafile and plots will be saved in the "outputs" file in the same directory as the ising_model.py file. The resulting plots for the measurements will appear on screen when the simulation and measurements are complete and will remain open at the end until the user closes it.
