# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:14:55 2026

@author: rohit
"""

import numpy as np
import matplotlib.pyplot as plt

#all constants
params = {
    'm': 1.0,      # kg
    'l': 0.2,      # m
    'kd': 0.5,     # m
    'g': 9.81,     # m/s^2
    'I': np.array([[0.01, 0, 0],
                   [0, 0.01, 0],
                   [0, 0, 0.01]])  # kg m^2
}

params['I_inv'] = np.linalg.inv(params['I'])

def quadcopter_derivatives(states, t, params, rotor_thrusts):
    #unpack state variables
    x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, wx, wy, wz = states
    
    #unpack thrusts
    T1, T2, T3, T4 = rotor_thrusts
    T = T1+T2+T3+T4
    
    RT = quaternion_to_rotation_matrix(q0, q1, q2, q3).T
    
    #translational dynamics (F=ma)
    grav_vec = np.array([0, 0, -params['m']*params['g']])  # force in N
    thrust_inertial = RT @ np.array([0, 0, T])              # force in N
    accel = (1/params['m']) * (grav_vec + thrust_inertial)
    x_ddot, y_ddot, z_ddot = accel
    
    #rotational dynamics
    omega = np.array([wx, wy, wz])
    
    tx = params['l'] * (T4+T3-T1-T2)
    ty = params['l'] * (T2-T1+T3-T4)
    tz = params['kd'] * (T1-T2+T3-T4)
    
    tau = np.array([tx, ty, tz])
    
    omega_dot = params['I_inv'] @ (tau - np.cross(omega, params['I'] @ omega))
    wx_dot, wy_dot, wz_dot = omega_dot
    
    q0_dot = 0.5*(-q1*wx - q2*wy - q3*wz)
    q1_dot = 0.5*(q0*wx+q2*wz-q3*wy)
    q2_dot = 0.5*(q0*wy-q1*wz+q3*wx)
    q3_dot = 0.5*(q0*wz+q1*wy-q2*wx)
    
    return np.array([
        x_dot,
        y_dot, 
        z_dot,
        x_ddot, 
        y_ddot,
        z_ddot,
        q0_dot, 
        q1_dot, 
        q2_dot,
        q3_dot, 
        wx_dot, 
        wy_dot,
        wz_dot
    ])

def quaternion_to_rotation_matrix(q0, q1, q2, q3):
    one = 1 - 2 * (q2**2 + q3**2)
    two = 2 * (q1*q2 - q0*q3)
    three = 2 * (q1*q3 + q0*q2)
    four = 2 * (q1*q2 + q0*q3)
    five = 1 - 2 * (q1**2 + q3**2)
    six = 2 * (q2*q3 - q0*q1)
    seven = 2 * (q1*q3 - q0*q2)
    eight = 2 * (q2*q3 + q0*q1)
    nine = 1 - 2 * (q1**2 + q2**2)
    return np.array([[one, two, three], [four, five, six], [seven, eight, nine]])

def rk4(state, t, dt, params, rotor_thrusts):
    k1 = quadcopter_derivatives(state, t, params, rotor_thrusts)
    k2 = quadcopter_derivatives(state + 0.5*dt*k1, t + 0.5*dt, params, rotor_thrusts)
    k3 = quadcopter_derivatives(state + 0.5*dt*k2, t + 0.5*dt, params, rotor_thrusts)
    k4 = quadcopter_derivatives(state + dt*k3, t + dt, params, rotor_thrusts)
    
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

### STATE VARIABLES AT t=0

#m
x_pos = 0
y_pos = 0
z_pos = 0

#m/s
x_dot_pos = 0
y_dot_pos = 0
z_dot_pos = 0

#dimensionless
q0 = 1
q1 = 0
q2 = 0
q3 = 0

#angular velocity- rad/s
wx =0
wy = 0
wz = 0

state = np.array([x_pos, y_pos, z_pos, x_dot_pos, y_dot_pos, z_dot_pos, q0, q1, q2, q3, wx, wy, wz])

rotor_thrusts = [2.2525, 2.2525, 2.6525, 2.6525]

t_start = 0 
t_end = 10
dt = 0.01
time_vector = np.arange(t_start, t_end + dt, dt)
state_hist = np.zeros((len(time_vector), 13))

#simulation loop
state_hist[0] = state  # store initial state
for i in range(len(time_vector) - 1):
    state_hist[i+1] = rk4(state_hist[i], time_vector[i], dt, params, rotor_thrusts)
    
    #normalization (accounts for quartneion drift)
    state_hist[i+1, 6:10] = state_hist[i+1, 6:10]/np.linalg.norm(state_hist[i+1, 6:10])

print("Initial state:", state_hist[0])
print("Final state:", state_hist[-1])
print("Quaternion norm at end:", np.linalg.norm(state_hist[-1, 6:10]))

#plotting!
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_hist[:, 0], state_hist[:, 1], state_hist[:, 2])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Quadrotor Trajectory')
plt.show()


#axes vs time graphs
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(time_vector, state_hist[:, 2])

axes[0].set_ylabel('Z (m)')
axes[1].plot(time_vector, state_hist[:, 0])

axes[1].set_ylabel('X (m)')
axes[2].plot(time_vector, state_hist[:, 1])

axes[2].set_ylabel('Y (m)')
axes[2].set_xlabel('Time (s)')
plt.suptitle('Position vs Time')
plt.tight_layout()
plt.show()