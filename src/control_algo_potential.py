# -*- coding: utf-8 -*-
"""
author: Sylvain Bertrand, 2023

   All variables are in SI units
    
   
   Variables used by the functions of this script
    - t: time instant (s)
    - robotNo: no of the current robot for which control is coputed (0 .. nbRobots-1)
    - poses:  size (3 x nbRobots)
        eg. of use: the pose of robot 'robotNo' can be obtained by: poses[:,robotNo]
            poses[robotNo,0]: x-coordinate of robot position (in m)
            poses[robotNo,1]: y-coordinate of robot position (in m)
            poses[robotNo,2]: orientation angle of robot (in rad)   (in case of unicycle dynamics only)
"""


import numpy as np
import math
from lib.potential import Potential
from lib.simulation import generate_init_positions


# ==============   "GLOBAL" VARIABLES KNOWN BY ALL THE FUNCTIONS ==============
# all variables declared here will be known by functions below
# use keyword "global" inside a function if the variable needs to be modified by the function

global firstCall   # can be used to check the first call ever of a function
firstCall = True

global pot # DO NOT MODIFY - allows initialisation of potential function from this script

# =============================================================================
def potential_seeking_ctrl(t, robotNo, robots_poses, _eval=False,_pot=None, difficulty=1, random=False):
# =============================================================================

        
    # --- example of modification of global variables ---
    # ---(updated values of global variables will be known at next call of this funtion) ---
    # global toto
    # firstCall = False
    
    global firstCall
    global pot
    
    # --- part to be run only once --- 
    if firstCall:
        if not _eval:
            pot = Potential(difficulty=difficulty, random=random)
        else:
            pot = _pot
        firstCall = False
    # --------------------------------
    
    # get number of robots
    N = robots_poses.shape[0]
    
    # get index of current robot  (short notation)
    i = robotNo

    # get positions of all robots (short notation)
    x = robots_poses[:,0:2]

    #get potential values measured by all robots at their current positions
    pot_measurement = np.zeros(N)
    for m in range(N):
        pot_measurement[m] = pot.value(x[m,:])

    # initialize control input vector
    ui = np.zeros(2)
    u0 = np.zeros(2) # for leader agent
    
    return ui[0], ui[1], pot   # potential is also returned to be used by main script for displays
# =============================================================================



