#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:01:35 2026

@author: gracetait
"""
import numpy as np
import matplotlib.pyplot as plt

class IsingModel(object):
    
    def __init__(self, n, T):
        # later add in the periodic boundary conditions!
        # also add in user input and whether or not to show the animation
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.T = T # thermal energy
        self.lattice = None
        
    def initialise(self):
        
        # Create a two-dimensional square lattice
        # Where -1 is down and 1 is up
        self.lattice = np.random.choice([-1,1], size = (self.n, self.n))
        
        return self.lattice
        
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
    
    def calculate_total_energy(self):
        # Sum of -J * Si * Sj for all unique neighbor pairs
        # Using np.roll allows us to calculate all "right" and "down" bonds efficiently
        # This avoids double-counting bonds.
        energy = - (np.sum(self.lattice * np.roll(self.lattice, 1, axis=0)) +
                    np.sum(self.lattice * np.roll(self.lattice, 1, axis=1)))
        
        return energy
    
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
            return self.lattice
        
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
    
    def calculate_magnetisation(self):
        
        # magnetisation is the sum of the spins divided by n
        # M = (spin ups + spin downs) / N^2 = np.mean
        return np.sum(self.lattice)
        
    #def calculate_specific_heat(self):
        
    
class Simulation(object):
    
    def __init__(self, n, T, dynamics):
        
        self.n = n
        self.T = T
        self.dynamics = dynamics    
        self.ising_model = IsingModel(n, T)
        self.ising_model.initialise()
        self.sweep = n * n # 1 sweep is n x n steps
        
        # empty arrays to hold data
        self.steps = []
        self.susceptibility = []
    
    def simulate(self, steps = 50000):
        
        fig, ax = plt.subplots()
    
        # Initialize the image object
        # vmin/vmax ensure -1 is black and 1 is white consistently
        im = ax.imshow(self.ising_model.lattice, cmap='binary', vmin=-1, vmax=1)
        
        # allow the user to choose the dynamics used
        if self.dynamics == "glauber":
            print("Glauber dynamics was chosen.")
            dynamics = self.ising_model.glauber_dynamics
            
        elif self.dynamics == "kawasaki":
            print("Kawasaki dynamics was chosen.")
            dynamics = self.ising_model.kawasaki_dynamics
            
        else:
            print("Dynamics not recognised.")
        
        
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
        
    def run_magnetisation_and_susceptibility(self, file):
        
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
            
            # make empty list for the new temperature
            m = []
            
            while counts < 10100 * N:
                
                # at each temperature, run the simulation
                model.glauber_dynamics()
                
                # add 1 to the counts
                counts += 1
                
                # need to wait 100 sweeps for equilibrium
                if counts < 100 * N:
                    continue
                
                # take a measurement every 10 sweeps
                if counts % (N * 10) == 0:
                    
                    # calculate magnetisation 
                    m.append(model.calculate_magnetisation())
                    
                    # convert to an array
                    mag = np.array(m)
                        
                    # calculate average magnetisation
                    # use absolute values to avoid M averaging to 0
                    avg_mag = np.mean(np.abs(mag))
                    
                    # calculate susceptibility where chi = 1/(N*T) * (<M^2> - <M>^2)
                    chi = (np.mean(mag**2) - np.mean(mag)**2) / (N * T)
                    #print(avg_mag, chi, T)
                    
                    # write the values into a file
                    f = open(file, "a")
                    f.write(f"{T}\n{avg_mag}\n{chi}\n")
                    f.close()
        
    def run_energy_and_specific_heat(self, file):
        
        # create a range for k_B T between 3 and 1 in steps of 0.1
        temperatures = np.arange(3.0, 0.9, -0.1)
        
        # generate an initial model to start with at the hottest T (3) 
        model = IsingModel(self.n, 3)
        model.initialise()
        
        # should this be inside or outside the loop?? (dont answer)
        # allow the user to choose the dynamics used
        if self.dynamics == "glauber":
            dynamics = model.glauber_dynamics
    
        elif self.dynamics == "kawasaki":
            dynamics = model.kawasaki_dynamics
            
        else:
            print("Dynamics not recognised.")
            return
        
        # calculate total energy 
        tot_E = model.calculate_total_energy()
        
        # number of total sites
        N = self.n**2
        
        # iterate through all temperatures
        for T in temperatures:
            print(f"Simulating T = {T:.1f}...")
            
            # make sure the model knows the new temperature
            model.T = T
            
            # start counts at zero
            counts = 0
            
            while counts < 10100 * N:
                
                # at each temperature, run the simulation
                lattice, del_E = dynamics()
            
                # add 1 to the counts
                counts += 1
            
                # need to wait 100 sweeps for equilibrium
                if counts < 100 * N:
                    continue
                
                # take a measurement every 10 sweeps
                if counts % (N * 10) == 0:
                    
                    # update the total energy using the energy difference
                    tot_E += del_E
                    
                    # calculate the average energy
                    avg_E = np.mean(tot_E)
    
                    # calculate specific heat
                    spec_he = (np.mean(tot_E**2) - np.mean(tot_E)) / (N * T**2)
                
                    # write the values into a file
                    f = open(file, "a")
                    f.write(f"{T}\n{avg_E}\n{spec_he}\n")
                    f.close()

    def plot_magnetisation_and_susceptibility(self, filename):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # create an empty list to store input data
        input_data = []        

        # read in the data from file
        filein = open(filename, "r")
        for line in filein.readlines():
            input_data.extend(line.strip(" \n").split(","))
        filein.close()
        
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
        ax1. plot(temperatures, magnetisation, 'o-', color='blue')
        ax1.set_ylabel("Average Total Magnetisation")
        ax1.set_title("Average Total Magnetisation Versus Thermal Energy")
        ax1.grid(True)
        
        # plot susceptibility
        ax2.plot(temperatures, susceptibility, 's-', color='red')
        ax2.set_xlabel("Thermal Energy (k_B T)")
        ax2.set_ylabel("Susceptibility (chi)")
        ax2.set_title("Susceptibility Versus Thermal Energy")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_energy_and_specific_heat(self, filename):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # create an empty list to store input data
        input_data = []        

        # read in the data from file
        filein = open(filename, "r")
        for line in filein.readlines():
            input_data.extend(line.strip(" \n").split(","))
        filein.close()
        
        # create emptry lists to hold temperature, magnetisation and susceptibility    
        temperatures = []
        total_energy = []
        specific_heat = []
        
        # iterate through input data and append to empty lists
        for i in range(0, len(input_data), 3):
            
            # obtain vlaue from input data
            temp = float(input_data[i])
            tot_e = float(input_data[i+1])
            spec_he = float(input_data[i+2])

            # append to empty lists
            temperatures.append(temp)
            total_energy.append(tot_e)
            specific_heat.append(spec_he)
        
        # plot average magnetisation
        ax1. plot(temperatures, total_energy, 'o-', color='blue')
        ax1.set_ylabel("Average Total Energy")
        ax1.set_title("Average Total Energy Versus Thermal Energy")
        ax1.grid(True)
        
        # plot susceptibility
        ax2.plot(temperatures, specific_heat, 's-', color='red')
        ax2.set_xlabel("Thermal Energy (k_B T)")
        ax2.set_ylabel("Heat Capacity per Spin")
        ax2.set_title("Heat Capacity per Spin Versus Thermal Energy")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
# let n be the lattice size
n = 50

# critical T is approx 2.26
T = 2

# choose the dynamics model 
ising_model_simulation = Simulation(n, T, dynamics = "glauber")
#ising_model_simulation = Simulation(n, T, dynamics = "kawasaki")

# to run the animation 
#ising_model_simulation.simulate() # glauber or kawasaki

# define file names to save data to
filename = "g_mag_sus_1.txt"
filename = "g_ene_spec_he_1.txt"
filename = "k_ene_spec_he_1.txt"

# to collect data (need to do this before plotting!)
ising_model_simulation.run_magnetisation_and_susceptibility(filename) # glauber
#ising_model_simulation.run_energy_and_specific_heat(filename) # glauber
#ising_model_simulation.run_energy_and_specific_heat(filename) # kawasaki

# to plot the graphs
ising_model_simulation.plot_magnetisation_and_susceptibility(filename) # only glauber
#ising_model_simulation.plot_energy_and_specific_heat(filename) # glauber or kawasaki
