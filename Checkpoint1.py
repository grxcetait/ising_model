#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:01:35 2026

@author: gracetait

Notes:
    - Kawasaki really not running correctly. Not sure what's wrong. Debug!
    - Need to write readme still
"""
import numpy as np
import matplotlib.pyplot as plt
import os

class IsingModel(object):
    """
    Class to define a two-dimensional Ising Model on a square (n x n) lattice.
    It simulates the behaviour of magnetic spins (up or down) using Monte Carlo methods.
    Glauber and Kawasaki dynamics can be chosen for the Monte Carlo simulation.
    """
    
    def __init__(self, n, T):
        """
        Initialises and defines parameters for the Ising Model square lattice.

        Parameters
        ----------
        n : int
            The length of the square lattice
        T : float
            The temperature of the system (represented as thermal energy)

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of spins in the square lattice
        self.T = T # Thermal energy
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
        self.lattice = np.random.choice([-1,1], size = (self.n, self.n))
        
        return self.lattice
    
    def glauber_dynamics(self):
        """
        Performs a single step of the Glauber (spin-flip) Monte Carlo simulation.
        This method picks a random site in the square lattice and decides whether to flip the spin
        based on the Metropolis algorithm. 

        Returns
        -------
        lattice : numpy.ndarray
            The updated 2D array of shape (n, n) representing the Ising Model square lattice
            with random spin state of up (+1) or down (-1).
        del_E = float
            The change in energy resulting from the single step of the Glauber (spin-flip) Monte Carlo simulation.
            Returns the actual energy difference if a flip occured, or 0 if the state remained the same.

        """
    
        # Choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(self.n)
        j = np.random.randint(self.n)
        
        # Obtain the current spin of the random site
        site_spin = self.lattice[i, j]
        
        # Calculate energy of the current and trial state
        E_current = self.calculate_energy(i, j, site_spin)
        
        # Calculate the change in energy between current and trial state
        del_E = - 2 * E_current
        
        # Flip the sign of the lattice site according to the Metropolis algorithm
        # Flip the sign if del_E is equal to or less than zero OR
        # If r is less than the boltzmann probability
        if del_E <= 0 or np.random.random() < np.exp(- del_E / self.T):
            
            # Flip the sign
            self.lattice[i,j] *= -1
            
            # Return the updated lattice and the change in energy
            return self.lattice, del_E
            
        # If we reach here, no flip happened, so return del_E = 0
        return self.lattice, 0
            
    
    def kawasaki_dynamics(self):
        """
        Performs a single step of the Kawasaki (spin-exchange) Monte Carlo simulation.
        This method picks two random sites in the square lattice and decides whether to exchange
        the spins based on the Metropolis algorithm.

        Returns
        -------
        lattice : numpy.ndarray
            The updated 2D array of shape (n, n) representing the Ising Model square lattice
            with random spin state of up (+1) or down (-1).
        del_E = float
            The change in energy resulting from the single step of the Kawasaki (spin-exchange) Monte Carlo simulation.
            Returns the actual energy difference if a swap occured, or 0 if the state remained the same.

        """
        
        # Choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(self.n)
        j = np.random.randint(self.n)
        
        # Choose another random site (k, m) in the lattice of size (n x n)
        k = np.random.randint(self.n)
        m = np.random.randint(self.n)
        
        # If both sites are the same, keep picking (k, m) until they are different
        while i == k and m == j:
            k = np.random.randint(self.n)
            m = np.random.randint(self.n)
            
        # Calculate the spins of both random sites
        site_ij_spin = self.lattice[i, j]
        site_km_spin = self.lattice[k, m]
        
        # If the spins are the same, swap does nothing
        if site_ij_spin == site_km_spin:
            return self.lattice, 0
        
        # Calculate energy of the current state
        E_current = self.calculate_energy(i, j, site_ij_spin) + self.calculate_energy(k, m, site_km_spin)
        
        # Calculate energy of the trial state
        E_trial = self.calculate_energy(i, j, site_km_spin) + self.calculate_energy(k, m, site_ij_spin)
        
        # Calculate the change in current and trial state
        del_E = E_trial - E_current
            
        # Calculate distance in x and y, considering periodic boundaries
        dx = abs(i - k)
        dy = abs(j - m)
        
        # If the distance is 1 (or n-1 for the wrap-around), they are neighbors
        is_neighbor = (dx == 0 and (dy == 1 or dy == self.n - 1)) or \
                      (dy == 0 and (dx == 1 or dx == self.n - 1))
        
        # If the sites are neighbours, the change in energy double counted the same site
        # Twice in the current and trial states
        # So, correct for it by adding 4 since the spins are always opposite 
        #if is_neighbor:
         #   del_E += 4 

        # Swap the signs of the lattice sites according to the Metropolis algorithm
        # Swap the signs if del_E is equal to or less than zero OR
        # If r is less than the boltzmann probability
        if del_E <= 0 or np.random.random() < np.exp(- del_E / self.T):
            
            # Swap the sites
            self.lattice[i, j] = site_km_spin
            self.lattice[k, m] = site_ij_spin
            
            # Return the updated lattice and the change in energy
            return self.lattice, del_E
                
        # If we reach here, no swap happened, so return del_E = 0
        return self.lattice, 0
            
    def calculate_energy(self, i, j, site_spin):
        """
        Calculates the energy of the site (i, j) with its four nearest neighbours.

        Parameters
        ----------
        i : int
            The row index of the site.
        j : int
            The column index of the site.
        site_spin : int
            The spin value up (+1) or down (-1) at the site

        Returns
        -------
        int
            The local energy calculated at the site (i, j) with its four nearest neighbours.

        """
    
        # Get the nearest neighbours of the site using modulus for periodic boundaries
        up = self.lattice[(i - 1) % self.n, j]
        down = self.lattice[(i + 1) % self.n, j]
        left = self.lattice[i, (j - 1) % self.n]
        right = self.lattice[i, (j + 1) % self.n]
    
        # Energy = - site * (sum of nearest neighbours)
        return - site_spin * (up + down + left + right)
    
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
        tot_E = - (np.sum(self.lattice * np.roll(self.lattice, 1, axis=0)) + 
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
    
    def __init__(self, n, T, dynamics):
        """
        Initialises and defines parameters for the simulation.

        Parameters
        ----------
        n : int
            The length of the square lattice
        T : float
            The temperature of the system (represented as thermal energy)
        dynamics : str
            The choice of the dynamics algorithm ('g' for Glauber or 'k' for Kawasaki)

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of spins in the square lattice (1 sweep)
        self.T = T # Thermal energy
        self.dynamics = dynamics # Choice of the dynamics algorithm
        self.ising_model = IsingModel(n, T) # Pass parameters into IsingModel
        self.ising_model.initialise() # Create the initial Ising Model square lattice
    
    def simulate(self, steps = 50100):
        """
        Runs an animation of the square lattice according to the chosen Monte
        Carlo simulation dynamics (Glauber or Kawasaki).

        Parameters
        ----------
        steps : int, optional
            The total number of Monte Carlo steps to run. The default is 50100.

        Returns
        -------
        None.

        """
        
        # Define the figure and axes for the animation 
        fig, ax = plt.subplots()
    
        # Initialize the image object
        # vmin/vmax ensure -1 is black and 1 is white consistently
        im = ax.imshow(self.ising_model.lattice, cmap='binary', vmin=-1, vmax=1)
        
        # Run the animation based on the dynamics chosen
        if self.dynamics == 'g':
            dynamics = self.ising_model.glauber_dynamics
            
        else:
            dynamics = self.ising_model.kawasaki_dynamics
        
        # Run the simulation for the total number of steps
        for s in range(steps):
            dynamics()

            # Update animation every 100 steps
            if s % 100 == 0:
                
                # Update the data in the existing plot
                im.set_data(self.ising_model.lattice)
                ax.set_title(f"Step: {s} | Temp: {self.T}")
                
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

        # Calculate and return specific heat
        return np.var(tot_E_list) / (self.N * self.T**2)
    
    def calculate_average_magnetisation(self, tot_M_list):
        """
        Calculates the mean absolute magnetisation based on the energy measurements.

        Parameters
        ----------
        tot_E_list : list or np.array
            A collection of total energy values recorded during the simulation
            after the system has reached equilibrium.

        Returns
        -------
        float
            The mean absolute magnetisation based on the energy measurements.

        """
        
        # Calcualte and return the mean absolute magnetisation
        return np.abs(np.mean(tot_M_list))
    
    def calculate_susceptibility(self, tot_M_list):
        """
        Calculates the susceptibility based on energy fluctuations from the 
        energy measurements.

        Parameters
        ----------
        tot_E_list : list or np.array
            A collection of total energy values recorded during the simulation
            after the system has reached equilibrium.

        Returns
        -------
        float
            The susceptibility based on energy fluctuations from the 
            energy measurements.

        """
        
        # Calculate and return the susceptibilty
        return np.var(np.abs(tot_M_list)) / (self.N * self.T)
    
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
             value = np.var(resample) / (self.N * self.T**2)
             
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
        temperatures = np.arange(3.0, 0.9, -0.1)
        
        # Generate an initial model to start with at the hottest T (3) 
        model = IsingModel(self.n, 3)
        model.initialise()
        
        # Iterate through all temperatures
        for T in temperatures:
            print(f"Simulating T = {T:.1f}...")
            
            # Make sure the model knows the new temperature
            model.T = T
            
            # Start counts at zero
            counts = 0
            
            # Make an empty list for the new temperature 
            tot_E_list = []
            
            # Only start calculating the total energy after equilibrium 
            tot_E = None
            
            # Iterate through 10000 sweeps after 100 sweeps of equilibrium
            while counts < 10100 * self.N:
                
                # At each temperature, run the simulation
                lattice, del_E = model.kawasaki_dynamics()
                
                # Add 1 to the counts
                counts += 1
            
                # Need to wait 100 sweeps for equilibrium
                if counts < 100 * self.N:
                    continue
                
                # Calculate the initial energy right after equilibrium
                if tot_E is None:
                    tot_E = model.calculate_total_energy()
                
                # Update the total energy using the energy difference
                else:
                    tot_E += del_E
                
                # Take a measurement every 10 sweeps
                if counts % (self.N * 10) == 0:
                    
                    # Append temperature to the list
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
        temperatures = np.arange(3.0, 0.9, -0.1)
        
        # Generate an initial model to start with at the hottest T (3) 
        model = IsingModel(self.n, 3)
        model.initialise()
        
        # Iterate through all temperatures
        for T in temperatures:
            print(f"Simulating T = {T:.1f}...")
            
            # Make sure the model knows the new temperature
            model.T = T
            
            # Start counts at zero
            counts = 0
            
            # Make an empty list for the new temperature 
            tot_E_list = []
            tot_M_list = []
            
            # Only start calculating the total energy after equilibrium 
            tot_E = None
            
            # Iterate through 10000 sweeps after 100 sweeps of equilibrium
            while counts < 10100 * self.N:
                
                # At each temperature, run the simulation
                lattice, del_E = model.glauber_dynamics()
                
                # Add 1 to the counts
                counts += 1
            
                # Need to wait 100 sweeps for equilibrium
                if counts < 100 * self.N:
                    continue
                
                # Calculate the initial energy right after equilibrium
                if tot_E is None:
                    tot_E = model.calculate_total_energy()
                
                # Update the total energy using the energy difference
                else:
                    tot_E += del_E
                
                # Take a measurement every 10 sweeps
                if counts % (self.N * 10) == 0:
                    
                    # Calculate magnetisation 
                    tot_M_list.append(model.calculate_magnetisation())
                    
                    # Append temperature to the list
                    tot_E_list.append(tot_E)
                    
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
        
        # Create emptry lists to hold temperature, magnetisation and susceptibility    
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
        ax2.set_ylabel("Heat Capacity per Spin")
        ax2.set_title("Heat Capacity per Spin Versus Thermal Energy")
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
            
            # If n is a positive integer, returnn
            if n > 0:
                return n
            
            # If not, prompt user again
            else:
                print("The system size (n) must be a positive integer. Please try again.")
            
        except ValueError:
            print("Please enter a valid integer.")
            
def temperature_prompt():
    """
    Prompts the user to enter a temperature value between 1 and 3.

    Returns
    -------
    T : float
        The temperature (T) provided by the user.

    """

    # Loop to prompt the user to enter a temperature value between 1 and 3
    while True:
        try:
            T = float(input("Enter temperature (T): "))
            
            # If T is a within 1 and 3, return T
            if T >= 1 and T <= 3:
                return T
            
            # If not, prompt user again
            else:
                print("The temperature must be a float between 1 and 3. Please try again.")
                
        except ValueError:
            print("The temperature must be a float between 1 and 3. Please try again.")
                
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
    

if __name__ == "__main__":
    
    # Promt the user for animation or measurements using a single character
    ani_or_mea = input("Enter 'a' for Animation or 'm' for Measurements: ")
    
    # If the correct characetrs were not chosen, promt the user to try again
    while ani_or_mea not in ['a', 'm']:
        
        print("Invalid entry. Please try again.")
        ani_or_mea = input("Enter 'a' for Animation or 'm' for Measurements: ")
    
    # For animation
    if ani_or_mea == 'a':
        
        print("Animation selected")
        
        # Prompt the user for animation parameters
        n = lattice_size_prompt()
        T = temperature_prompt()
        dynamics = dynamics_prompt()
        
        # Initialise and run the simulation
        sim = Simulation(n, T, dynamics)
        sim.simulate()
    
    # For measurements
    elif ani_or_mea == 'm':
        
        print("Measurements selected")
        
        # Prompt the user for measurement parameters
        n = lattice_size_prompt()
        T = 0 # since measurements option doesn't need a temperature
        dynamics = dynamics_prompt()
        
        # Initialise the simulation
        sim = Simulation(n, T, dynamics)
        
        # Run the simulation using Glauber dynamics
        if dynamics == 'g':
            
            # Print messages
            print("Glauber dynamics was chosen.")
            print("Measurements will be taken for the average energy, average absolute value of the magnetisation,")
            print("specific heat and susceptibility.")
            print("Measurements beginning...")
            
            # Define filenames
            filename1 = "glauber_energy_and_specific_heat_5.txt"
            filename2 = "glauber_magnetisation_and_susceptibility_5.txt"
            
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
            filename = "kawasaki_energy_and_specific_heat_5.txt"
            
            # Run the simulation
            sim.run_kawasaki(filename)
            
            # Plot the data
            sim.plot_energy_and_specific_heat(filename)