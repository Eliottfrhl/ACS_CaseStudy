# -*- coding: utf-8 -*-
"""
===============================================================
 Multi-Agent Potential Seeking — Student Simulation Script
===============================================================
 Course  : EDC 3A — Multi-Agent Systems
 Author  : Sylvain Bertrand, 2025
 Updated : Aarsh Thakker, 2026
---------------------------------------------------------------
 OBJECTIVE:
   A fleet of robots must cooperatively find the maximum of
   an unknown potential field defined over a 2D environment.

 YOUR TASK:
   Implement your control algorithm in:
       --> control_algo_potential.py

 HOW TO RUN:
   $ python etude_de_cas_prof.py

 ROBOT DYNAMICS (choose in SECTION 1):
   - singleIntegrator2D : state=[x,y],      input=[vx, vy]
   - unicycle           : state=[x,y,theta], input=[V, omega]

     Helper: [V, omega] = si_to_uni([vx,vy], theta, kp=gain)
===============================================================
"""

# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from lib.simulation import FleetSimulation, generate_init_positions
from lib.robot import Fleet, si_to_uni
import control_algo_potential
import eval_metrics

# ===============================================================
# SECTION 1 — SIMULATION PARAMETERS
#             Modify this section to configure your simulation
# ===============================================================

# --- Number of robots ---
nbOfRobots = 5
# --- Simulation duration (seconds) ---
Tsim = 30.0
# --- Sampling period (seconds) — do not change ---
Ts = 0.05
DIFFICULTY = 3  # 1 (easy) to 3 (hard) — controls the shape of the potential field
RANDOM = True  # Set to False for non-deterministic randomisation
# --- Robot dynamics model ---
# Options: 'singleIntegrator2D'  or  'unicycle'
robotDynamics = 'singleIntegrator2D'

# --- Display console output during simulation? ---
DISP_CONSOLE = False  # Set to True to see potential values at each timestep (can slow down the simulation)

# --- Show plots at the end? ---
SHOW_ANIMATION   = True    # live animation (press ESC to skip)
SHOW_TRAJECTORY  = False    # 2D trajectory plot
SHOW_STATES      = False    # state components vs time
SHOW_CONTROLS    = False    # control inputs vs time
SHOW_POTENTIAL   = True    # potential measurements vs time

# ---------------------------------------------------------------
# >>> CHOOSE YOUR INITIALISATION MODE HERE <<<
# ---------------------------------------------------------------
initPositions = generate_init_positions(
    nbOfRobots,
    mode    = 'grid',      # 'grid' | 'random' | 'manual'
    center  = (-20, -20),
    spacing = 1.0,
    # data  = [[-20,-20],[-19,-20],[-18,-20],[-17,-20],[-16,-20]]  # for 'manual'
)

# ===============================================================
# SECTION 2 — FLEET & SIMULATION SETUP
#             No changes needed here
# ===============================================================

# Build initial poses for unicycle (adds orientation angle theta)
if robotDynamics == 'unicycle':
    initAngles = np.zeros((nbOfRobots, 1))
    initPoses  = np.concatenate((initPositions, initAngles), axis=1)

# Create fleet
if robotDynamics == 'singleIntegrator2D':
    fleet = Fleet(nbOfRobots, dynamics=robotDynamics, initStates=initPositions)
else:
    fleet = Fleet(nbOfRobots, dynamics=robotDynamics, initStates=initPoses)

# Create simulation object
simulation = FleetSimulation(fleet, t0=0.0, tf=Tsim, dt=Ts)

# History of potential values measured by each robot at each timestep
# Shape: (nb_timesteps, nbOfRobots)
potential_measurements = np.zeros((simulation.t.shape[0], nbOfRobots))
t_index = 0

# ===============================================================
# SECTION 3 — SIMULATION LOOP
#             No changes needed here
# ===============================================================

print('\n' + '='*70)
print('  SIMULATION STARTING')
print(f'  Robots: {nbOfRobots}   |   Duration: {Tsim}s   |   Dynamics: {robotDynamics}')
print('='*70)

if DISP_CONSOLE:
    print('\n  [   potential value measured by each robot   ] | max value to be found')
    print('  ' + '-'*68)

for t in simulation.t:

    # --- Get current poses of all robots (nb_robots x nb_states) ---
    robots_poses = fleet.getPosesArray()

    # --- Compute control input for each robot ---
    for robotNo in range(fleet.nbOfRobots):

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  YOUR ALGORITHM IS CALLED HERE
        #  Implement potential_seeking_ctrl() in control_algo_potential.py
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vx, vy, pot = control_algo_potential.potential_seeking_ctrl(
            t, robotNo, robots_poses, difficulty=DIFFICULTY, random=RANDOM
        )

        if robotDynamics == 'singleIntegrator2D':
            fleet.robot[robotNo].ctrl = np.array([vx, vy])
        else:
            fleet.robot[robotNo].ctrl = si_to_uni(
                vx, vy, robots_poses[robotNo, 2], kp=10.
            )

    # --- Store potential measurements ---
    current_measurements = pot.value(robots_poses[:, 0:2])
    potential_measurements[t_index, :] = current_measurements

    # --- Console output ---
    if DISP_CONSOLE and t_index > 0:
        max_val = np.max(pot.value(pot.mu))
        print(f'  {current_measurements}  |  {max_val:.4f}')

    # --- Sanity check on potential values ---
    if -10 in potential_measurements[t_index, :]:
        print('\n  ⚠️  ERROR: Invalid potential value (-10) detected!')
        print('  Check your Potential parameters or restart with a new random seed.\n')

    t_index += 1

    # --- Advance simulation ---
    simulation.addDataFromFleet(fleet)
    fleet.integrateMotion(Ts)

print('\n' + '='*70)
print('  SIMULATION COMPLETE')
print('='*70 + '\n')


# ===============================================================
# SECTION 4 — EVALUATION METRICS
#             Automatic scoring of your algorithm
# ===============================================================

relative_pot_found_error, total_distance = eval_metrics.eval_metrics(
    simulation, potential_measurements, pot
)

max_val_found  = np.max(potential_measurements)
max_val_target = np.max(pot.value(pot.mu))

print('  ┌─────────────────────────────────────────┐')
print('  │           EVALUATION RESULTS            │')
print('  ├─────────────────────────────────────────┤')
print(f'  │  Max potential (target)  : {max_val_target:>10.4f}   │')
print(f'  │  Max potential (found)   : {max_val_found:>10.4f}   │')
print(f'  │  Localisation error      : {relative_pot_found_error:>10.4f}   │')
print(f'  │  Total distance traveled : {total_distance:>9.2f} m  │')
print('  └─────────────────────────────────────────┘\n')


# ===============================================================
# SECTION 5 — PLOTS
#             Visualise the results
# ===============================================================

colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

# --- Live animation ---
if SHOW_ANIMATION:
    print('  Showing animation... (press ESC to close and continue)\n')
    fig1, ax1 = pot.plot(1)
    simulation.animation(
        figNo=1, potential=pot, pause=0.001,
        robot_scale=0.2, xmin=-25, xmax=25, ymin=-25, ymax=25
    )

# --- 2D trajectories ---
if SHOW_TRAJECTORY:
    simulation.plotXY(
        figNo=2, potential=pot, xmin=-25, xmax=25, ymin=-25, ymax=25
    )

# --- State components vs time ---
if SHOW_STATES:
    simulation.plotState(figNo=3)

# --- Control inputs vs time ---
if SHOW_CONTROLS:
    simulation.plotCtrl(figNo=6)

# --- Potential measurements vs time ---
if SHOW_POTENTIAL:
    plt.figure(figsize=(10, 4))
    for rr in range(fleet.nbOfRobots):
        plt.plot(
            simulation.t, potential_measurements[:, rr],
            color=colorList[rr % len(colorList)],
            label=f'Robot {rr}'
        )
    plt.axhline(
        y=max_val_target, color='k', linestyle='--', linewidth=1.5,
        label=f'Max to find ({max_val_target:.3f})'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Potential value (-)')
    plt.title('Potential measurements per robot over time')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

plt.show()