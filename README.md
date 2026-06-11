## Overview
Nonlinear 6-DOF quadrotor dynamics simulation built from first principles in Python. Implements Newton-Euler equations of motion with quaternion kinematics, RK4 numerical integration, a cascaded PID controller, and an LQR controller linearized around hover.

## Physics
Translational Dyanmics
mẍ = [0 0 -mg] + R^T [0 0 T]

Rotational Dynamics
## Overview
Nonlinear 6-DOF quadrotor dynamics simulation built from first principles in Python. Implements Newton-Euler equations of motion with quaternion kinematics, RK4 numerical integration, a cascaded PID controller, and an LQR controller linearized around hover.

## Physics
Translational Dyanmics
mẍ = [0 0 -mg] + R^T [0 0 T]

Rotational Dynamics
Iω̇  = T_total - w x (Iw)

Quaternion 
q̇ = q/2 ⊗ [0 w]

RK4 Integration
xn+1 = xn + delta t/t (k1 + 2k2 + 2k3 + k4)

## Simplifying Assumptions
Rigid body, no aerodynamic drag, no wind, no rotor dynamics (instantaneous thrust response), flat earth, small angle approximation for LQR linearization only

## Controllers
PID Controller - Two loops, inner loop controls attitude and outer loop controlls altitude. Uses hand tuned gains and error states to compute rotor thrusts.
There are 6 controllers in a cascade - altitude, roll, pitch, yaw, x-position, y-position - fed into a mixer that solves thrust allocation.
LQR Controller - Linear Quadratic Regulator. Solves an optimization problem given two human made constraits, one that penalizes state error and another that penalizes control effort. The controller solves the opitmization problem and controls the craft accordingly. 

## Results
There are 3 sims, all of them output 2 things, a 3-d path plot, and 3 state vs time positon plots
The 3 sims are - an uncontrolled quadcopter sim, a PID controlled sim, and a LQR sim.
File names: uncontrolled dynamics.py, PID controlled dynamics.py, and LQR Controlled.py

## How to Run
Ensure numpy, scipy, and matplotlib are installed.
Install files and run through spyder.

Quaternion 
q̇ = q/2 ⊗ [0 w]

RK4 Integration
xn+1 = xn + delta t/t (k1 + 2k2 + 2k3 + k4)

## Simplifying Assumptions
Rigid body, no aerodynamic drag, no wind, no rotor dynamics (instantaneous thrust response), flat earth, small angle approximation for LQR linearization only

## Controllers
PID Controller - Two loops, inner loop controls attitude and outer loop controlls altitude. Uses hand tuned gains and error states to compute rotor thrusts.
There are 6 controllers in a cascade - altitude, roll, pitch, yaw, x-position, y-position - fed into a mixer that solves thrust allocation.
LQR Controller - Linear Quadratic Regulator. Solves an optimization problem given two human made constraits, one that penalizes state error and another that penalizes control effort. The controller solves the opitmization problem and controls the craft accordingly. 

## Results
There are 3 sims, all of them output 2 things, a 3-d path plot, and 3 state vs time positon plots
The 3 sims are - an uncontrolled quadcopter sim, a PID controlled sim, and a LQR sim.
File names: uncontrolled dynamics.py, PID controlled dynamics.py, and LQR Controlled.py

## How to Run
Ensure numpy, scipy, and matplotlib are installed.
Install files and run through spyder.
