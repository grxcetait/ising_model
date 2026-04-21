#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:01:35 2026

@author: gracetait
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from numba import njit

@njit
def calculate_local_energy(i, j, n, lattice, site_spin, J):
    """
    Calculates the local energy of site (i, j) with its four nearest neighbours.

    Arguments:
        i: row index of the site
        j: column index of the site
        n: system size
        lattice: spin lattice
        site_spin: spin value at site (i, j)
        J: coupling constant

    Returns:
        local energy at site (i, j) with its four nearest neighbours
    """

    # Get the nearest neighbours of the site using modulus for periodic boundaries
    up = lattice[(i - 1) % n, j]
    down = lattice[(i + 1) % n, j]
    left = lattice[i, (j - 1) % n]
    right = lattice[i, (j + 1) % n]

    # Energy = -J * site * (sum of nearest neighbours)
    return - J * site_spin * (up + down + left + right)


@njit
def glauber_sweep(lattice, n, N, kbT, J):
    """
    Performs one full sweep (N steps) of the Glauber (spin-flip) Monte Carlo simulation.
    Runs entirely in compiled code to avoid Python overhead.

    Arguments:
        lattice: spin lattice
        n: system size
        N: number of steps per sweep (= n * n)
        kbT: thermal energy
        J: coupling constant

    Returns:
        total energy change over the sweep
    """

    # Initialise total energy change for this sweep
    del_E_total = 0.0

    # Perform N individual Glauber steps
    for _ in range(N):

        # Choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

        # Obtain the current spin of the random site
        site_spin = lattice[i, j]

        # Calculate energy of the current state and the change in energy upon flipping
        E_current = calculate_local_energy(i, j, n, lattice, site_spin, J)
        del_E = - 2.0 * E_current

        # Flip the spin according to the Metropolis algorithm
        if del_E <= 0:

            # Always accept energy-lowering flip
            lattice[i, j] *= -1
            del_E_total += del_E

        elif np.random.random() < np.exp(- del_E / kbT):

            # Accept energy-raising flip with Boltzmann probability
            lattice[i, j] *= -1
            del_E_total += del_E

    return del_E_total


@njit
def kawasaki_sweep(lattice, n, N, kbT, J):
    """
    Performs one full sweep (N steps) of the Kawasaki (spin-exchange) Monte Carlo simulation.
    Runs entirely in compiled code to avoid Python overhead.

    Arguments:
        lattice: spin lattice
        n: system size
        N: number of steps per sweep (= n * n)
        kbT: thermal energy
        J: coupling constant

    Returns:
        total energy change over the sweep
    """

    # Initialise total energy change for this sweep
    del_E_total = 0.0

    # Perform N individual Kawasaki steps
    for _ in range(N):

        # Choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

        # Choose another random site (k, m) in the lattice of size (n x n)
        k = np.random.randint(0, n)
        m = np.random.randint(0, n)

        # If both sites are the same, keep picking (k, m) until they are different
        while i == k and m == j:
            k = np.random.randint(0, n)
            m = np.random.randint(0, n)

        # Calculate the spins of both random sites
        site_ij_spin = lattice[i, j]
        site_km_spin = lattice[k, m]

        # If the spins are the same, swap does nothing
        if site_ij_spin == site_km_spin:
            continue

        # Calculate energy of the current state
        E_current = (calculate_local_energy(i, j, n, lattice, site_ij_spin, J) +
                     calculate_local_energy(k, m, n, lattice, site_km_spin, J))

        # Calculate energy of the trial state
        E_trial = (calculate_local_energy(i, j, n, lattice, site_km_spin, J) +
                   calculate_local_energy(k, m, n, lattice, site_ij_spin, J))

        # Calculate the change in energy between current and trial state
        del_E = E_trial - E_current

        # Calculate distance in x and y, considering periodic boundaries
        dx = abs(i - k)
        dy = abs(j - m)

        # If the sites are neighbours, the change in energy double counted the same site
        # Twice in the current and trial states
        # So, correct for it by adding 4*J since the spins are always opposite
        is_neighbor = (dx == 0 and (dy == 1 or dy == n - 1)) or \
                      (dy == 0 and (dx == 1 or dx == n - 1))

        if is_neighbor:
            del_E += 4.0 * J

        # Swap the spins according to the Metropolis algorithm
        if del_E <= 0:

            # Always accept energy-lowering swap
            lattice[i, j] = site_km_spin
            lattice[k, m] = site_ij_spin
            del_E_total += del_E

        elif np.random.random() < np.exp(- del_E / kbT):

            # Accept energy-raising swap with Boltzmann probability
            lattice[i, j] = site_km_spin
            lattice[k, m] = site_ij_spin
            del_E_total += del_E

    return del_E_total


class IsingModel(object):
    """
    Class to define a two-dimensional Ising Model on a square (n x n) lattice.
    It simulates the behaviour of magnetic spins (up or down) using Monte Carlo methods.
    Glauber and Kawasaki dynamics can be chosen for the Monte Carlo simulation.
    """

    def __init__(self, n, kbT, ordering, J=1):
        """
        Initialises and defines parameters for the Ising Model square lattice.

        Parameters
        ----------
        n : int
            The length of the square lattice
        kbT : float
            The thermal energy of the system
        ordering : str
            The choice of the initial lattice to be ordered ('o') or disordered ('d')
        J : float, optional
            The coupling constant. The default is 1.

        Returns
        -------
        None.

        """

        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of spins in the square lattice
        self.kbT = kbT # Thermal energy
        self.ordering = ordering # Choice of the initial lattice to be ordered or disordered
        self.J = J # Coupling constant
        self.lattice = None

    def initialise(self):
        """
        Creates the initial configuration of the Ising Model on a square (n x n) lattice by
        assigning a random spin state where -1 is down and +1 is up, to every site.

        Returns
        -------
        lattice : numpy.ndarray
            A 2D array of shape (n, n) representing the initial configuration
            of the lattice, with random spin states of up (+1) or down (-1)

        """

        # Create a two-dimensional square lattice
        # Where -1 is down and 1 is up
        # Create an ordered or disordered lattice based on user input
        # Use int32 for better numba performance
        if self.ordering == 'd': # For a disordered initial lattice
            self.lattice = np.random.choice([-1, 1], size=(self.n, self.n)).astype(np.int32)

        else: # For an ordered initial lattice with all spins up
            self.lattice = np.ones((self.n, self.n), dtype=np.int32)

        #return self.lattice

    def calculate_magnetisation(self):
        """
        Calculates the total magnetisation of the current Ising Model square lattice state.

        Returns
        -------
        int
            The total amgnetisation of the lattice, representing the sum of all
            of the individual spin values of up (+1) or down (-1).

        """

        # Magnetisation is the sum of the spins
        return np.sum(self.lattice)

    def calculate_total_energy(self):
        """
        Calculates the total energy of the current Ising Model square lattice state.

        Returns
        -------
        tot_E : int
            The total energy of the current Ising Model square lattice state.

        """

        # Sum of -J * Si * Sj for all unique neighbor pairs
        # Using np.roll allows us to avoids double-counting bonds
        tot_E = - self.J * (np.sum(self.lattice * np.roll(self.lattice, 1, axis=0)) +
                    np.sum(self.lattice * np.roll(self.lattice, 1, axis=1)))

        # Here, np.roll is used by shifting the lattice by one unit horizontally and vertically
        # If the sites have the same spins, they equal +1
        # If the sites have different spins, they equal -1
        # By rolling ones per axis, we include periodic boundaries as and
        # Prevents double counting as pairs are checked between sites only once

        return tot_E


class Simulation(object):
    """
    A class designed to manage the Ising Model simulations.
    It manages the Ising Model square lattice created by IsingModel and
    executes Monte Carlo Glauber or Kawasaki dynamics.
    An animation can be ran or measurements taken for thermodynamic properties,
    and saves datafiles and plots.
    """

    def __init__(self, n, kbT, steps, ordering, dynamics, J=1):
        """
        Initialises and defines parameters for the simulation.

        Parameters
        ----------
        n : int
            The length of the square lattice
        kbT : float
            The thermal energy of the system
        steps : int
            The number of steps for the animation
        ordering : str
            The choice of the initial lattice to be ordered ('o') or disordered ('d')
        dynamics : str
            The choice of the dynamics algorithm ('g' for Glauber or 'k' for Kawasaki)
        J : float, optional
            The coupling constant. The default is 1.

        Returns
        -------
        None.

        """

        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of spins in the square lattice (1 sweep)
        self.kbT = kbT # Thermal energy
        self.steps = steps # Choice of number of steps for the animation
        self.dynamics = dynamics # Choice of the dynamics algorithm
        self.ordering = ordering # Choice of the initial lattice to be ordered or disordered
        self.J = J # Coupling constant

    def simulate(self):
        """
        Runs an animation of the square lattice according to the chosen Monte
        Carlo simulation dynamics (Glauber or Kawasaki).

        Returns
        -------
        None.

        """

        # Initialise the lattice using the IsingModel class
        ising_model = IsingModel(self.n, self.kbT, self.ordering, self.J) # Pass parameters into IsingModel
        ising_model.initialise() # Create the initial Ising Model square lattice

        # Define the figure and axes for the animation
        fig, ax = plt.subplots()

        # Initialize the image object
        # vmin/vmax ensure -1 is black and 1 is white consistently
        im = ax.imshow(ising_model.lattice, cmap='binary', vmin=-1, vmax=1)

        # Select the compiled sweep function based on the chosen dynamics
        if self.dynamics == 'g':
            sweep = glauber_sweep
        else:
            sweep = kawasaki_sweep

        # Run the simulation for N^2 total sweeps to show the well-equilibrated end result
        total_sweeps = self.N

        # Run the simulation for the total number of sweeps
        for s in range(total_sweeps):

            # Run one full sweep using the compiled function
            sweep(ising_model.lattice, self.n, self.N, self.kbT, self.J)

            # Update animation every 10 sweeps
            if s % 10 == 0:

                # Update the data in the existing plot
                im.set_data(ising_model.lattice)
                ax.set_title(f"Sweep: {s} | Thermal Energy: {self.kbT}")

                # Keep the image up while the script is running
                plt.pause(0.001)

        # Keep the final image open when the loop finishes
        plt.show()

    def calculate_average_energy(self, tot_E_list):
        """
        Calculates the mean energy per of the current Ising Model square lattice state.

        Parameters
        ----------
        tot_E_list : list or np.array
            A collection of total energy values recorded during the simulation
            after the system has reached equilibrium.

        Returns
        -------
        float
            The mean of the energy measurements.

        """

        # Calculate and return the mean energy
        return np.mean(tot_E_list)

    def calculate_specific_heat(self, tot_E_list):
        """
        Calculates the specific heat based on energy fluctuations from the
        energy measurements.

        Parameters
        ----------
        tot_E_list : list or np.array
            A collection of total energy values recorded during the simulation
            after the system has reached equilibrium.

        Returns
        -------
        float
            The specific heat based on the energy measurements from the
            energy measurements.

        """

        #return np.var(tot_E_list) / (self.N * self.T**2)

        # Convert to numpy array
        tot_E_list = np.array(tot_E_list)

        # Calculate and return specific heat
        mean_E_squared = np.mean(tot_E_list**2)
        mean_E_squared_avg = np.mean(tot_E_list)**2
        return (mean_E_squared - mean_E_squared_avg) / (self.N * self.kbT**2)

    def calculate_average_magnetisation(self, tot_M_list):
        """
        Calculates the mean absolute magnetisation based on the magnetisation measurements.

        Parameters
        ----------
        tot_M_list : list or np.array
            A collection of total magnetisation values recorded during the simulation
            after the system has reached equilibrium.

        Returns
        -------
        float
            The mean absolute magnetisation based on the magnetisation measurements.

        """

        # Calculate and return the mean absolute magnetisation
        return np.mean(np.abs(tot_M_list))

    def calculate_susceptibility(self, tot_M_list):
        """
        Calculates the susceptibility based on energy fluctuations from the
        energy measurements.

        Parameters
        ----------
        tot_M_list : list or np.array
            A collection of total magnetisation values recorded during the simulation
            after the system has reached equilibrium.

        Returns
        -------
        float
            The susceptibility based on energy fluctuations from the
            energy measurements.

        """

        #return np.var(np.abs(tot_M_list)) / (self.N * self.T) # this uses the absolute value but that is wrong
        #return np.var(tot_M_list) / (self.N * self.T) # calcualte the susceptibility not using the absolute value

        # Convert to numpy array
        tot_M_list = np.array(tot_M_list)

        # Calculate and return the susceptibilty
        mean_M_squared = np.mean(tot_M_list**2)
        mean_M = np.mean(tot_M_list)
        return (mean_M_squared - mean_M**2) / (self.N * self.kbT)

    def bootstrap_method(self, data):
        """
        Calculates the errors for the specific heat based on the Bootstrap
        method which is a statistical resampling technique.

        Parameters
        ----------
        data : list or np.ndarray
            The energy measurements collected after the system reached equilibrium.

        Returns
        -------
        float
            The standard error of the specific heat calculation.

        """

        # Convert to numpy array
        data = np.array(data)

        # Find the length of the data
        n = len(data)

        # Create an empty list to store the resampled values
        resampled_values = []

        # Resampling 1000 times is sufficient
        for j in range(1000):

             # Randomnly resample from the n measurements
             ind = np.random.randint(0, n, size = n)
             resample = data[ind]

             # Calculate specific heat accordingly
             #value = np.var(resample) / (self.N * self.kbT**2)
             mean_E_sq = np.mean(resample**2)
             mean_E = np.mean(resample)**2
             value = (mean_E_sq - mean_E) / (self.N * self.kbT**2)

             # Append to the list
             resampled_values.append(value)

        # Calculate and return the error
        # Which is the standard deviation of the resampled values
        return np.std(np.array(resampled_values))

    def run_kawasaki(self, filename):
        """
        Runs an automated simulation using Kawasaki dynamics across a range of
        temperatures.

        It starts from T = 3.0 to T = 1.0, decreasing by steps of 0.1. For each
        temperature, the system is allowed to reach thermal equilibrium before
        taking periodic measurements of the energy and heat capacity. The data
        is saved to a specified text file.

        Parameters
        ----------
        filename : str
            The name of the text file where the temperature, average energy,
            specific heat, and specific heat error will be stored.

        Returns
        -------
        None.

        """

        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)

        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)

        # Create a range for k_B T between 3 and 1 in steps of 0.1
        temperatures = np.round(np.arange(3.0, 0.9, -0.1), 1)

        # Initialise the lattice using the IsingModel class and start at the hottest T (3)
        ising_model = IsingModel(self.n, 3, self.ordering, self.J) # Pass parameters into IsingModel
        ising_model.initialise() # Create the initial Ising Model square lattice

        # Warm up the numba function with a single call before the main loop
        kawasaki_sweep(ising_model.lattice, self.n, 1, 3.0, self.J)

        # Iterate through all temperatures
        for T in temperatures:
            print(f"Simulating kbT = {T:.1f}...")

            # Make sure the model and this class knows the new temperature
            ising_model.kbT = T
            self.kbT = T

            # Make an empty list for the new temperature
            tot_E_list = []

            # Equilibrate the system for 200 sweeps using the compiled sweep function
            for _ in range(200):
                kawasaki_sweep(ising_model.lattice, self.n, self.N, T, self.J)

            # Calculate the total energy right after equilibrium
            tot_E = ising_model.calculate_total_energy()

            # Take 1000 measurements, one every 10 sweeps
            for _ in range(1000):

                # Run 10 sweeps between measurements using the compiled sweep function
                for _ in range(10):
                    del_E = kawasaki_sweep(ising_model.lattice, self.n, self.N, T, self.J)

                    # Update the total energy using the energy difference to avoid recalculation
                    tot_E += del_E

                # Append the current total energy measurement
                tot_E_list.append(tot_E)

            # After completing all the measurements for the temperature
            # Calculate the average energy and specific heat
            avg_E = self.calculate_average_energy(tot_E_list)
            spec_he = self.calculate_specific_heat(tot_E_list)
            spec_he_err = self.bootstrap_method(tot_E_list)

            # Write the values into the specified file
            with open(file_path, "a") as f:
                f.write(f"{T},{avg_E},{spec_he},{spec_he_err}\n")

    def run_glauber(self, filename1, filename2):
        """
        Runs an automated simulation using Glauber dynamics across a range of
        temperatures.

        It starts from T = 3.0 to T = 1.0, decreasing by steps of 0.1. For each
        temperature, the system is allowed to reach thermal equilibrium before
        taking periodic measurements of the energy, heat capacity, magnetisation
        and susceptibility. The data is saved to the specified text files

        Parameters
        ----------
        filename1 : str
            The name of the text file where the temperature, average energy,
            specific heat, and specific heat error will be stored.
        filename2 : str
            The name of the text file where the temperature, average magnetisation,
            and susceptibility will be stored.

        Returns
        -------
        None.

        """

        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file1_path = os.path.join(datafiles_folder, filename1)
        file2_path = os.path.join(datafiles_folder, filename2)

        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)

        # Create a range for k_B T between 3 and 1 in steps of 0.1
        temperatures = np.round(np.arange(3.0, 0.9, -0.1), 1)

        # Initialise the lattice using the IsingModel class and start at the hottest T (3)
        ising_model = IsingModel(self.n, 3, self.ordering, self.J) # Pass parameters into IsingModel
        ising_model.initialise() # Create the initial Ising Model square lattice

        # Warm up the numba function with a single call before the main loop
        glauber_sweep(ising_model.lattice, self.n, 1, 3.0, self.J)

        # Iterate through all temperatures
        for T in temperatures:
            print(f"Simulating kbT = {T:.1f}...")

            # Make sure the model this class knows the new temperature
            ising_model.kbT = T
            self.kbT = T

            # Make an empty list for the new temperature
            tot_E_list = []
            tot_M_list = []

            # Equilibrate the system for 200 sweeps using the compiled sweep function
            for _ in range(200):
                glauber_sweep(ising_model.lattice, self.n, self.N, T, self.J)

            # Calculate the total energy right after equilibrium
            tot_E = ising_model.calculate_total_energy()

            # Take 1000 measurements, one every 10 sweeps
            for _ in range(1000):

                # Run 10 sweeps between measurements using the compiled sweep function
                for _ in range(10):
                    del_E = glauber_sweep(ising_model.lattice, self.n, self.N, T, self.J)

                    # Update the total energy using the energy difference to avoid recalculation
                    tot_E += del_E

                # Append the current total energy and magnetisation measurements
                tot_E_list.append(tot_E)
                tot_M_list.append(ising_model.calculate_magnetisation())

            # After completing all the measurements for the temperature
            # Calculate the average energy and specific heat
            avg_E = self.calculate_average_energy(tot_E_list)
            spec_he = self.calculate_specific_heat(tot_E_list)
            spec_he_err = self.bootstrap_method(tot_E_list)

            # Calculate average magnetisation and susceptibility
            avg_mag = self.calculate_average_magnetisation(tot_M_list)
            chi = self.calculate_susceptibility(tot_M_list)

            # Write the values into the specified file
            with open(file1_path, "a") as f:
                f.write(f"{T},{avg_E},{spec_he},{spec_he_err}\n")

            # Write the values into the specified file
            with open(file2_path, "a") as f:
                f.write(f"{T},{avg_mag},{chi}\n")

    def plot_magnetisation_and_susceptibility(self, filename):
        """
        Read the saved text files and produces png plots for magnetisation and
        susceptibility versus temperature.

        Parameters
        ----------
        filename : str
            The name of the text file where the temperature, average magnetisation,
            and susceptibility are stored.

        Returns
        -------
        None.

        """

        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")

        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create empty plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # Create an empty list to store input data
        input_data = []

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))

        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")

        # Create empty lists to hold temperature, magnetisation and susceptibility
        temperatures = []
        magnetisation = []
        susceptibility = []

        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 3):

            # Obtain vlaue from input data
            temp = float(input_data[i])
            mag = float(input_data[i+1])
            sus = float(input_data[i+2])

            # Append to empty lists
            temperatures.append(temp)
            magnetisation.append(mag)
            susceptibility.append(sus)

        # Plot average magnetisation
        ax1.plot(temperatures, magnetisation, 'o-', color='blue',
                 markerfacecolor = 'black', markeredgecolor = 'black',
                 markersize = 4)
        ax1.set_ylabel("Average Total Magnetisation")
        ax1.set_title("Average Total Magnetisation Versus Thermal Energy")
        ax1.grid(True)

        # Plot susceptibility
        ax2.plot(temperatures, susceptibility, 'o-', color='red',
                 markerfacecolor = 'black', markeredgecolor = 'black',
                 markersize = 4)
        ax2.set_xlabel("Thermal Energy (k_B T)")
        ax2.set_ylabel("Susceptibility (chi)")
        ax2.set_title("Susceptibility Versus Thermal Energy")
        ax2.grid(True)

        # Fix any overlapping labels, titles or tick marks
        plt.tight_layout()

        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)

        # Print message
        print(f"Plot successfully saved to: {save_path}")

        # Show final plots
        plt.show()

    def plot_energy_and_specific_heat(self, filename):
        """
        Read the saved text files and produces png plots for energy and specific
        heat versus temperature.

        Parameters
        ----------
        filename : str
            The name of the text file where the temperature, average energy,
            specific heat, and specific heat error are stored.

        Returns
        -------
        None.

        """

        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")

        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create empty plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # Create an empty list to store input data
        input_data = []

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))

        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")

        # Create emptry lists to hold temperature, magnetisation and susceptibility
        temperatures = []
        total_energy = []
        specific_heat = []
        specific_heat_error =[]

        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 4):

            # Obtain vlaue from input data
            temp = float(input_data[i])
            tot_e = float(input_data[i+1])
            spec_he = float(input_data[i+2])
            spec_he_err = float(input_data[i+3])

            # Append to empty lists
            temperatures.append(temp)
            total_energy.append(tot_e)
            specific_heat.append(spec_he)
            specific_heat_error.append(spec_he_err)

        # Plot average magnetisation
        ax1.plot(temperatures, total_energy, 'o-', color='blue',
                 markerfacecolor = 'black', markeredgecolor = 'black',
                 markersize = 4)
        ax1.set_ylabel("Average Total Energy")
        ax1.set_title("Average Total Energy Versus Thermal Energy")
        ax1.grid(True)

        # Plot susceptibility
        ax2.errorbar(temperatures, specific_heat, yerr = specific_heat_error,
                 fmt='o-', color='red', ecolor='black', markerfacecolor = 'black', markeredgecolor = 'black',
                 capsize=3, elinewidth=1, markeredgewidth=1, markersize = 4)
        ax2.set_xlabel("Thermal Energy (k_B T)")
        ax2.set_ylabel("Specific Heat (c)")
        ax2.set_title("Specific Heat Versus Thermal Energy")
        ax2.grid(True)

        # Fix any overlapping labels, titles or tick marks
        plt.tight_layout()

        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)

        # Print message
        print(f"Plot successfully saved to: {save_path}")

        # Show final plots
        plt.show()


def lattice_size_prompt():
    """
    Prompts the suer to enter a positive integer for the Ising Model square
    lattice system size.

    Returns
    -------
    n : int
        The system size (n) provided by the user.

    """

    # Loop to promt the user to enter a positive integer for n
    while True:
        try:
            n = int(input("Enter system size (n): "))

            # If n is a positive integer, return
            if n > 0:
                return n

            # If not, prompt user again
            else:
                print("The system size (n) must be a positive integer. Please try again.")

        except ValueError:
            print("Please enter a valid integer.")

def thermal_energy_prompt():
    """
    Prompts the user to enter a thermal energy value between 1 and 3.

    Returns
    -------
    T : float
        The temperature (T) provided by the user.

    """

    # Loop to prompt the user to enter a thermal energy value between 1 and 3
    while True:
        try:
            kbT = float(input("Enter temperature (T): "))

            # If T is a within 1 and 3, return T
            #if kbT >= 1 and kbT <= 3:
                #return kbT

            # If not, prompt user again
            #else:
                #print("The thermal energy must be a float between 1 and 3. Please try again.")
            return kbT

        except ValueError:
            #print("The thermal energy must be a float between 1 and 3. Please try again.")
            print("Invalid. Please try again.")

def dynamics_prompt():
    """
    Prompts the user to choose between Glauber ('g') ir Kawasaki ('k') dynamics.

    Returns
    -------
    dynamics : str
        A single character ('g' or 'k') representing the chosen dynamics.

    """

    # Promt the user to enter a single character to choose the dynamics
    dynamics = input("Enter the desired dynamics to be used, 'g' for Glauber or 'k' for Kawasaki: ")

    # If the correct charactesr were not chosen, prompt the user to try again
    while dynamics not in ['g', 'k']:
        print("Dynamics not recognised. Please try again.")
        dynamics = input("Enter the desired dynamics to be used, 'g' for Glauber or 'k' for Kawasaki: ")

    return dynamics

def animation_steps_prompt():
    """
    Prompts the user to input the number of steps for the animation

    Returns
    -------
    steps : int
        The number of steps for the animation chosen by the user.

    """

    # Loop to promt the user to enter a positive integer for the number of steps for the animation
    while True:
        try:
            steps = int(input("Enter the number of steps for the animation: "))

            # If n is a positive integer, return
            if steps > 0:
                return steps

            # If not, prompt user again
            else:
                print("The number of steps must be a positive integer. Please try again.")

        except ValueError:
            print("Please enter a valid integer.")

def initialise_state():

    # Promt the user to enter a single character to choose the ordering of the initial state
    ordering = input("Enter the desired ordering of the initial state, 'o' for ordered or 'd' for disordered: ")

    # If the correct character was not chosen, prompt the user to try again
    while ordering not in ['o', 'd']:
        print("Initial state not recognised. Please try again.")
        ordering = input("Enter the desired ordering of the initial state, 'o' for ordered or 'd' for disordered: ")

    return ordering


if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='2D Ising Model Monte Carlo simulation')
    parser.add_argument('-n', '--size', type=int, default=50,
                        help='System size (default: 50)')
    parser.add_argument('-T', '--temperature', type=float, default=2.0,
                        help='Thermal energy kbT, used for animation mode (default: 2.0)')
    parser.add_argument('-J', '--coupling', type=float, default=1,
                        help='Coupling constant (default: 1)')
    parser.add_argument('-d', '--dynamics', type=str, choices=['g', 'k'], default='g',
                        help="Dynamics type: 'g' for Glauber, 'k' for Kawasaki (default: 'g')")
    parser.add_argument('-o', '--ordering', type=str, choices=['o', 'd'], default='d',
                        help="Initial lattice ordering: 'o' for ordered, 'd' for disordered (default: 'd')")
    parser.add_argument('-m', '--mode', type=str, choices=['a', 'm'], default='a',
                        help="Run mode: 'a' for animation, 'm' for measurements (default: 'a')")
    args = parser.parse_args()

    # For animation
    if args.mode == 'a':

        print("Animation selected")

        # Initialise and run the simulation
        sim = Simulation(args.size, args.temperature, 0, args.ordering, args.dynamics, args.coupling)
        sim.simulate()

    # For measurements
    elif args.mode == 'm':

        print("Measurements selected")

        # Initialise the simulation
        sim = Simulation(args.size, 3, 0, args.ordering, args.dynamics, args.coupling)

        # Run the simulation using Glauber dynamics
        if args.dynamics == 'g':

            # Print messages
            print("Glauber dynamics was chosen.")
            print("Measurements will be taken for the average energy, average absolute value of the magnetisation,")
            print("specific heat and susceptibility.")
            print("Measurements beginning...")

            # Define filenames
            filename1 = "glauber_energy_and_specific_heat_final.txt"
            filename2 = "glauber_magnetisation_and_susceptibility_final.txt"

            # Run the simulation
            sim.run_glauber(filename1, filename2)

            # Plot the data
            sim.plot_energy_and_specific_heat(filename1)
            sim.plot_magnetisation_and_susceptibility(filename2)

        # Run the simulation using Kawasaki dynamics
        else:

            # Print messages
            print("Kawasaki dynamics was chosen.")
            print("Measurements will be taken for the average energy and specific heat.")
            print("Measurements beginning...")

            # Define filename
            filename = "kawasaki_energy_and_specific_heat_final.txt"

            # Run the simulation
            sim.run_kawasaki(filename)

            # Plot the data
            sim.plot_energy_and_specific_heat(filename)

def plot_electric_field_measurements(self, filename):

    # Define datafiles output directory
    base_directory = os.path.dirname(os.path.abspath(__file__))
    outputs_directory = os.path.join(base_directory, "outputs")
    filename_path = os.path.join(outputs_directory, "datafiles", filename)
    plots_folder = os.path.join(outputs_directory, "plots")

    # If the folders dont exist, create them
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Create an empty list to store input data
    input_data = []

    # Read in the data from the specified text file
    try:
        with open(filename_path, "r") as filein:
            for line in filein:
                input_data.append(line.strip("\n").split(","))

    # If text file cannot be found, print error
    except FileNotFoundError:
        print(f"Error: Could not find {filename_path}")

    # Create an empty lattice
    E_x = np.zeros(shape = (self.l, self.l, self.l))
    E_y = np.zeros(shape = (self.l, self.l, self.l))
    E_z = np.zeros(shape = (self.l, self.l, self.l))
    E = np.zeros(shape = (self.l, self.l, self.l))

    # Convert input data into a np array
    input_data = np.array(input_data[1:], dtype = float)

    # Collect the input data
    x = input_data[:, 4].astype(int)
    y = input_data[:, 5].astype(int)
    z = input_data[:, 6].astype(int)
    E_x[x, y, z] = input_data[:, 0]
    E_y[x, y, z] = input_data[:, 1]
    E_z[x, y, z] = input_data[:, 2]
    E[x, y, z] = input_data[:, 3]

    # Extract the midplanes
    E_x_midplane = E_x[:, :, self.l // 2]
    E_y_midplane = E_y[:, :, self.l // 2]

    # Create a meshgrid
    X, Y = np.meshgrid(np.arange(self.l), np.arange(self.l))

    # Skip every 5 pixels so the arrows aren't too crowded
    skip = (slice(None, None, 5), slice(None, None, 5))

    # Calculate the magnitude of the electric field in the x and y plane
    mag = np.sqrt(E_x_midplane**2 + E_y_midplane**2)

    # Create empty plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Plot the potential
    plot = ax.quiver(X[skip], Y[skip], (E_x_midplane / mag).T[skip],
                     (E_y_midplane / mag).T[skip], cmap = "viridis")
    ax.imshow(mag.T, origin="lower", cmap="viridis", alpha=0.4,
      extent=[0, self.l, 0, self.l])
    plt.colorbar(plot, label = "Field Magnitude $|E|$")
    ax.set_title(r"Electric Field Vectors $E_x, E_y$", fontsize = 16)
    ax.set_aspect("equal")

    # Save the plots to the plots folder
    save_filename = filename.replace(".txt", "_plot.png")
    save_path = os.path.join(plots_folder, save_filename)
    plt.savefig(save_path, dpi = 300)

    # Print message
    print(f"Plots successfully saved to: {save_path}")

    # Show final plot
    plt.show()