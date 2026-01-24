#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:01:35 2026

@author: gracetait
"""
import numpy as np
import matplotlib.pyplot as plt
import os

class IsingModel(object):
    
    def __init__(self, n, T):
        # later add in the periodic boundary conditions!
        # also add in user input and whether or not to show the animation
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # number of squares in lattice
        self.T = T # thermal energy
        self.lattice = None
        
    def initialise(self):
        
        # Create a two-dimensional square lattice
        # Where -1 is down and 1 is up
        self.lattice = np.random.choice([-1,1], size = (self.n, self.n))
        
        return self.lattice
    
    def glauber_dynamics(self):
    
        # choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(self.n)
        j = np.random.randint(self.n)
        
        # obtain the current spin of the random site
        site_spin = self.lattice[i, j]
        
        # calculate energy of the current and trial state
        E_current = self.calculate_energy(i, j, site_spin)
        E_trial = self.calculate_energy(i, j, -site_spin)
        
        # calculate the change in energy between current and trial state
        del_E = E_trial - E_current
        
        # flip the sign of the lattice site according to the energy change
        # if del_E is equal to or less than zero, flip the spin
        if del_E <= 0:
            
            # flip the sign
            self.lattice[i,j] *= -1
            
        # otherwise, flip the spin with boltzmann probability
        else: 
            # generate a random number r in [0, 1]
            r = np.random.random() # is this correct?
            
            # if r is less than the boltzmann factor, flip the sign
            # otherwise, do not flip the sign    
            if r < np.exp(- del_E / self.T):
                
                # flip the sign
                self.lattice[i,j] *= -1
                
        return self.lattice, del_E
    
    def kawasaki_dynamics(self):
        
        # choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(self.n)
        j = np.random.randint(self.n)
        
        # choose another random site (k, m) in the lattice of size (n x n)
        k = np.random.randint(self.n)
        m = np.random.randint(self.n)
        
        # if both sites are the same, keep picking (k, m) until they are different
        while i == k and m == j:
            k = np.random.randint(self.n)
            m = np.random.randint(self.n)
            
        # calculate the spins of both random sites
        site_ij_spin = self.lattice[i, j]
        site_km_spin = self.lattice[k, m]
        
        # if the spins are the same, swap does nothing
        if site_ij_spin == site_km_spin:
            return self.lattice, 0
        
        # calculate energy of the current state
        E_current = self.calculate_energy(i, j, site_ij_spin) + self.calculate_energy(k, m, site_km_spin)
        
        # calculate energy of the trial state
        E_trial = self.calculate_energy(i, j, site_km_spin) + self.calculate_energy(k, m, site_ij_spin)
        
        # calcualte the change in current and trial state
        del_E = E_trial - E_current
            
        # calculate distance in x and y, considering periodic boundaries
        dx = abs(i - k)
        dy = abs(j - m)
        
        # still don't really understand this so go over it! (dont answer)
        # if the distance is 1 (or n-1 for the wrap-around), they are neighbors
        is_neighbor = (dx == 0 and (dy == 1 or dy == self.n - 1)) or \
                      (dy == 0 and (dx == 1 or dx == self.n - 1))
        
        # dont understand why there is a 4! look over this again (don't answer)
        if is_neighbor:
            del_E += 4 * site_ij_spin * site_km_spin
            
        # flip the sign of the lattice site according to the energy change
        # if del_E is equal to or less than zero, flip the spin
        if del_E <= 0:
            
            # swap the sites
            self.lattice[i, j] = site_km_spin
            self.lattice[k, m] = site_ij_spin
            
        # otherwise, flip the spin with boltzmann probability
        else: 
            # generate a random number r in [0, 1]
            r = np.random.random() # is this correct?
            
            # if r is less than the boltzmann factor, flip the sign
            # otherwise, do not flip the sign    
            if r < np.exp(- del_E / self.T):
                
                # swap the sites
                self.lattice[i, j] = site_km_spin
                self.lattice[k, m] = site_ij_spin
                
        return self.lattice, del_E
            
    # Check if i did the indicies correct?!
    def calculate_energy(self, i, j, site_spin):
    # Is this the same for glauber and kawasaki dynamics?
    
        # get the nearest neighbours of the site using modulus for periodic boundaries
        up = self.lattice[(i - 1) % self.n, j]
        down = self.lattice[(i + 1) % self.n, j]
        left = self.lattice[i, (j - 1) % self.n]
        right = self.lattice[i, (j + 1) % self.n]
    
        # energy = - site * (sum of nearest neighbours)
        return - site_spin * (up + down + left + right)
    
    def calculate_magnetisation(self):
        
        # magnetisation is the sum of the spins divided by n
        # M = (spin ups + spin downs) / N^2 = np.mean
        return np.sum(self.lattice)
    
    def calculate_total_energy(self):
        # Sum of -J * Si * Sj for all unique neighbor pairs
        # Using np.roll allows us to calculate all "right" and "down" bonds efficiently
        # This avoids double-counting bonds.
        tot_E = - (np.sum(self.lattice * np.roll(self.lattice, 1, axis=0)) +
                    np.sum(self.lattice * np.roll(self.lattice, 1, axis=1)))
        
        return tot_E
    
    
    
    
    
class Simulation(object):
    
    def __init__(self, n, T, dynamics):
        
        self.n = n
        self.T = T
        self.dynamics = dynamics    
        self.ising_model = IsingModel(n, T)
        self.ising_model.initialise()
        self.N = n * n # 1 sweep is n x n steps
        
        # empty arrays to hold data
        self.steps = []
        self.susceptibility = []
    
    def simulate(self, steps = 50000):
        
        fig, ax = plt.subplots()
    
        # Initialize the image object
        # vmin/vmax ensure -1 is black and 1 is white consistently
        im = ax.imshow(self.ising_model.lattice, cmap='binary', vmin=-1, vmax=1)
        
        # allow the user to choose the dynamics used
        if self.dynamics == 'g':
            dynamics = self.ising_model.glauber_dynamics
            
        else:
            dynamics = self.ising_model.kawasaki_dynamics
        
        for s in range(steps):
            dynamics()
            
            # need to understand why it is self.ising_model.lattice
            # update animation every 100 steps
            if s % 100 == 0:
                # Update the data in the existing plot
                im.set_data(self.ising_model.lattice)
                ax.set_title(f"Step: {s} | Temp: {self.T}")
                
                # This is the "magic" line. It opens the window, 
                # draws the update, and keeps the script running.
                plt.pause(0.001)
    
        # At the very end, call plt.show() normally 
        # This keeps the final window open when the loop finishes.
        plt.show()
        
    def calculate_average_energy(self, tot_E_list):
        
        # calculate the average energy
        return np.mean(tot_E_list)
    
    def calculate_specific_heat(self, tot_E_list):

        # calculate specific heat
        return np.var(tot_E_list) / (self.N * self.T**2)
    
    def calculate_average_magnetisation(self, tot_M_list):
        
        # can't remember if tyler said np.abs is good or bad?
        return np.abs(np.mean(tot_M_list))
    
    def calculate_susceptibility(self, tot_M_list):
        
        # is np.abs good here?
        return np.var(np.abs(tot_M_list)) / (self.N * self.T)
    
    def bootstrap_method(self, data):
        
        # convert to numpy array
        data = np.array(data)
        
        n = len(data)
        resampled_values = []
        
        # resampling 1000 times is sufficient
        for j in range(1000):
            
             # randomnly resample from the n measurements
             ind = np.random.randint(0, n, size = n)
             resample = data[ind]
             
             # calculate specific heat accordingly
             value = np.var(resample) / (self.N * self.T**2)
             
             # append to the list
             resampled_values.append(value)
        
        # calculate and return error which is the standard deviation of the resampled values
        return np.std(np.array(resampled_values))
        
    def run_kawasaki(self, filename):
        
        # define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # if the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # create a range for k_B T between 3 and 1 in steps of 0.1
        temperatures = np.arange(3.0, 0.9, -0.1)
        
        # generate an initial model to start with at the hottest T (3) 
        model = IsingModel(self.n, 3)
        model.initialise()
        
        # number of total sites
        N = self.n**2
        
        # iterate through all temperatures
        for T in temperatures:
            print(f"Simulating T = {T:.1f}...")
            
            # make sure the model knows the new temperature
            model.T = T
            
            # start counts at zero
            counts = 0
            
            # make an empty list for the new temperature 
            tot_E_list = []
            
            # calculate total energy 
            tot_E = model.calculate_total_energy()
            
            while counts < 10100 * N:
                
                # at each temperature, run the simulation
                lattice, del_E = model.kawasaki_dynamics()
                
                # add 1 to the counts
                counts += 1
            
                # need to wait 100 sweeps for equilibrium
                if counts < 100 * N:
                    continue
                
                # take a measurement every 10 sweeps
                if counts % (N * 10) == 0:
                    
                    # update the total energy using the energy difference
                    tot_E += del_E
                    
                    # append temperature to the list
                    tot_E_list.append(tot_E)
                    
            # after completing all the measurements for the temperature
            # calculate the average energy and specific heat
            avg_E = self.calculate_average_energy(tot_E_list)
            spec_he = self.calculate_specific_heat(tot_E_list)
            spec_he_err = self.bootstrap_method(tot_E_list)
         
            # write the values into a file
            with open(file_path, "a") as f:
                f.write(f"{T},{avg_E},{spec_he},{spec_he_err}\n")
                
    def run_glauber(self, filename1, filename2):
        
        # define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file1_path = os.path.join(datafiles_folder, filename1)
        file2_path = os.path.join(datafiles_folder, filename2)
        
        # if the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # create a range for k_B T between 3 and 1 in steps of 0.1
        temperatures = np.arange(3.0, 0.9, -0.1)
        
        # generate an initial model to start with at the hottest T (3) 
        model = IsingModel(self.n, 3)
        model.initialise()
        
        # number of total sites
        N = self.n**2
        
        # iterate through all temperatures
        for T in temperatures:
            print(f"Simulating T = {T:.1f}...")
            
            # make sure the model knows the new temperature
            model.T = T
            
            # start counts at zero
            counts = 0
            
            # make an empty list for the new temperature 
            tot_E_list = []
            tot_M_list = []
            
            # calculate total energy 
            tot_E = model.calculate_total_energy()
            
            while counts < 10100 * N:
                
                # at each temperature, run the simulation
                lattice, del_E = model.glauber_dynamics()
                
                # add 1 to the counts
                counts += 1
            
                # need to wait 100 sweeps for equilibrium
                if counts < 100 * N:
                    continue
                
                # take a measurement every 10 sweeps
                if counts % (N * 10) == 0:
                    
                    # calculate magnetisation 
                    tot_M_list.append(model.calculate_magnetisation())
                    
                    # update the total energy using the energy difference
                    tot_E += del_E
                    
                    # append temperature to the list
                    tot_E_list.append(tot_E)
                    
            # after completing all the measurements for the temperature
            # calculate the average energy and specific heat
            avg_E = self.calculate_average_energy(tot_E_list)
            spec_he = self.calculate_specific_heat(tot_E_list)
            spec_he_err = self.bootstrap_method(tot_E_list)
            
            # calculate average magnetisation and susceptibility
            avg_mag = self.calculate_average_magnetisation(tot_M_list)
            chi = self.calculate_susceptibility(tot_M_list)
         
            # write the values into a file
            with open(file1_path, "a") as f:
                f.write(f"{T},{avg_E},{spec_he},{spec_he_err}\n")
                
            # write the values into a file
            with open(file2_path, "a") as f:
                f.write(f"{T},{avg_mag},{chi}\n")

    def plot_magnetisation_and_susceptibility(self, filename):
        
        # define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")

        # if the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # create an empty list to store input data
        input_data = []        

        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
        
        # create emptry lists to hold temperature, magnetisation and susceptibility    
        temperatures = []
        magnetisation = []
        susceptibility = []
        
        # iterate through input data and append to empty lists
        for i in range(0, len(input_data), 3):
            
            # obtain vlaue from input data
            temp = float(input_data[i])
            mag = float(input_data[i+1])
            sus = float(input_data[i+2])
            
            # append to empty lists
            temperatures.append(temp)
            magnetisation.append(mag)
            susceptibility.append(sus)
        
        # plot average magnetisation
        ax1.plot(temperatures, magnetisation, 'o-', color='blue', 
                 markerfacecolor = 'black', markeredgecolor = 'black',
                 markersize = 4)
        ax1.set_ylabel("Average Total Magnetisation")
        ax1.set_title("Average Total Magnetisation Versus Thermal Energy")
        ax1.grid(True)
        
        # plot susceptibility
        ax2.plot(temperatures, susceptibility, 'o-', color='red', 
                 markerfacecolor = 'black', markeredgecolor = 'black',
                 markersize = 4)
        ax2.set_xlabel("Thermal Energy (k_B T)")
        ax2.set_ylabel("Susceptibility (chi)")
        ax2.set_title("Susceptibility Versus Thermal Energy")
        ax2.grid(True)
        
        plt.tight_layout()
        
        # save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        print(f"Plot successfully saved to: {save_path}")
        plt.show()
    
    def plot_energy_and_specific_heat(self, filename):
        
        # define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # if the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # create empty plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # create an empty list to store input data
        input_data = []      
        
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
        
        # create emptry lists to hold temperature, magnetisation and susceptibility    
        temperatures = []
        total_energy = []
        specific_heat = []
        specific_heat_error =[]
        
        # iterate through input data and append to empty lists
        for i in range(0, len(input_data), 4):
            
            # obtain vlaue from input data
            temp = float(input_data[i])
            tot_e = float(input_data[i+1])
            spec_he = float(input_data[i+2])
            spec_he_err = float(input_data[i+3])

            # append to empty lists
            temperatures.append(temp)
            total_energy.append(tot_e)
            specific_heat.append(spec_he)
            specific_heat_error.append(spec_he_err)
        
        # plot average magnetisation
        ax1.plot(temperatures, total_energy, 'o-', color='blue', 
                 markerfacecolor = 'black', markeredgecolor = 'black',
                 markersize = 4)
        ax1.set_ylabel("Average Total Energy")
        ax1.set_title("Average Total Energy Versus Thermal Energy")
        ax1.grid(True)
        
        # plot susceptibility
        ax2.errorbar(temperatures, specific_heat, yerr = specific_heat_error, 
                 fmt='o-', color='red', ecolor='black', markerfacecolor = 'black', markeredgecolor = 'black',
                 capsize=3, elinewidth=1, markeredgewidth=1, markersize = 4)
        ax2.set_xlabel("Thermal Energy (k_B T)")
        ax2.set_ylabel("Heat Capacity per Spin")
        ax2.set_title("Heat Capacity per Spin Versus Thermal Energy")
        ax2.grid(True)
        
        plt.tight_layout()
        
        # save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        print(f"Plot successfully saved to: {save_path}")
        
        # show plots
        plt.show()
    
    

def lattice_size_prompt():

    while True:
        try:
            n = int(input("Enter system size (n): "))
            if n > 0:
                return n
        except ValueError:
            print("Please enter a valid integer.")
            
def temperature_promt():

    while True:
        try:
            T = float(input("Enter temperature (T): "))
            if T >= 1 and T <= 3:
                return T
            
            else:
                print("The temperature must be a float between 1 and 3. Please try again.")
                
        except ValueError:
            print("The temperature must be a float between 1 and 3. Please try again.")
                
def dynamics_prompt():
    
    dynamics = input("Enter the desired dynamics to be used, 'g' for Glauber or 'k' for Kawasaki: ")
    
    while dynamics not in ['g', 'k']:
        print("Dynamics not recognised. Please try again.")
        dynamics = input("Enter the desired dynamics to be used, 'g' for Glauber or 'k' for Kawasaki: ")
        
    return dynamics
    

if __name__ == "__main__":
    
    # promt for required values
    ani_or_mea = input("Enter 'a' for Animation or 'm' for Measurements: ")
    
    while ani_or_mea not in ['a', 'm']:
        
        print("Invalid entry. Please try again.")
        ani_or_mea = input("Enter 'a' for Animation or 'm' for Measurements: ")
    
    if ani_or_mea == 'a':
        
        print("Animation selected")
        
        # prompts
        n = lattice_size_prompt()
        T = temperature_promt()
        dynamics = dynamics_prompt()
        
        # initialise
        sim = Simulation(n, T, dynamics)
        sim.simulate()
    
    elif ani_or_mea == 'm':
        
        print("Measurements selected")
        
        # promts
        n = lattice_size_prompt()
        T = temperature_promt()
        dynamics = dynamics_prompt()
        
        # initialise 
        sim = Simulation(n, T, dynamics)
        
        if dynamics == 'g':
            print("Glauber dynamics was chosen.")
            print("Measurements will be taken for the average energy, average absolute value of the magnetisation,")
            print("specific heat and susceptibility.")
            print("Measurements beginning...")
            
            # define filenames
            filename1 = "glauber_energy_and_specific_heat_1.txt"
            filename2 = "glauber_magnetisation_and_susceptibility_2.txt"
            
            # run the simulation
            sim.run_glauber(filename1, filename2)
            
            # plot the data
            sim.plot_energy_and_specific_heat(filename1)
            sim.plot_magnetisation_and_susceptibility(filename2)
            
            
        else:
            print("Kawasaki dynamics was chosen.")
            print("Measurements will be taken for the average energy and specific heat.")
            print("Measurements beginning...")
            
            # define filename
            filename = "kawasaki_energy_and_specific_heat_1.txt"
            
            # run the simulation
            sim.run_kawasaki(filename)
            
            # plot the data
            sim.plot_energy_and_specific_heat(filename)