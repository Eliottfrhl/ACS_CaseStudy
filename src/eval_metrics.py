# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:06:00 2025

@author: S. Bertrand
"""

import numpy as np

# Compute metrics


def eval_metrics(simulation, potential_measurements, pot):

    distances = np.zeros(simulation.nbOfRobots)
    
    
    for i_rob in range(simulation.nbOfRobots):
        simu = simulation.robotSimulation[i_rob]
        nb_times = len(simu.t)
        for i_time in range(1,nb_times):
            distances[i_rob] += np.linalg.norm(simu.state[i_time] - simu.state[i_time-1])
    
    
    max_potentials = np.max(potential_measurements, axis=0)
    
    max_pot_to_be_found = np.max(pot.value(pot.mu))
    max_pot_found = np.max(max_potentials)
    
    
    relative_pot_found_error = np.abs((max_pot_to_be_found-max_pot_found) / max_pot_to_be_found  )
    
    total_distance = np.sum(distances)


    return relative_pot_found_error, total_distance
