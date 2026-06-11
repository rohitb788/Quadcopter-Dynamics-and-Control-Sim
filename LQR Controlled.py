# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:14:55 2026

@author: rohit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

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

A = np.zeros((12, 12))
A[0, 3] = 1
A[1, 4] = 1
A[2, 5] = 1
A[3, 7] = -params['g']
A[4, 6] = -params['g']
A[6, 9] = 1
A[7, 10] = 1
A[8, 11] = 1

m = params['m']
Ixx = params['I'][0, 0]
Iyy = params['I'][1, 1]
Izz = params['I'][2, 2]
kd = params['kd']
l = params['l']

B = np.zeros((12, 4))
B[5, 0] = B[5, 1] = B[5, 2] = B[5, 3] = 1/m
B[9, 0] = B[9, 1] = -l/Ixx
B[9, 2] = B[9, 3] = l/Ixx
B[10, 0] = B[10, 3] = -l/Iyy
B[10, 1] = B[10, 2] = l/Iyy
B[11, 0] = B[11, 2] = kd/Izz
B[11, 1] = B[11, 3] = -kd/Izz

Q = np.diag([1, 1, 1,      # x, y, z — reduced
             1, 1, 1,       # velocities
             5, 5, 5,       # angles — increased
             0.1, 0.1, 0.1]) # angular rates

R = np.eye(4) * 5.0        # much more conservative # wx, wy, wz


#solving for K using 
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

params['I_inv'] = np.linalg.inv(params['I'])

def quaternion_to_euler(q0, q1, q2, q3):
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    theta = np.arcsin(2*(q0*q2 - q3*q1))
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    
    return phi, theta, psi

def lqr_controller(state, state_desired, K, params):
    m = params['m']
    g = params['g']
    
    # unpack 13-element quaternion state
    x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, wx, wy, wz = state
    
    # convert quaternion to euler angles
    phi, theta, psi = quaternion_to_euler(q0, q1, q2, q3)
    
    # build 12-element LQR state
    lqr_state = np.array([x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, wx, wy, wz])
    
    # compute state error
    x_error = lqr_state - state_desired
    
    # hover thrust bias — 4 element vector
    u_hover = np.array([m*g/4, m*g/4, m*g/4, m*g/4])
    
    # control law
    u = -K @ x_error + u_hover
    
    u = np.clip(u, 0, 20)  # physical rotor limits
    return u
    

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

# desired state
state_desired = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# initial state — small perturbation from desired
x_pos, y_pos, z_pos = 0, 0, 4.8        # 0.2m below desired
x_dot_pos, y_dot_pos, z_dot_pos = 0, 0, 0
q0, q1, q2, q3 = 1, 0, 0, 0
wx, wy, wz = 0, 0, 0.1                  # small yaw rate disturbance

state = np.array([x_pos, y_pos, z_pos, x_dot_pos, y_dot_pos, z_dot_pos, q0, q1, q2, q3, wx, wy, wz])

rotor_thrusts = [2.2525, 2.2525, 2.6525, 2.6525]

t_start = 0 
t_end = 10
dt = 0.01
time_vector = np.arange(t_start, t_end + dt, dt)
state_hist = np.zeros((len(time_vector), 13))

#simulation loop
state_hist[0] = state
for i in range(len(time_vector) - 1):
    # get rotor thrusts from LQR
    rotor_thrusts = lqr_controller(state_hist[i], state_desired, K, params)
    
    # rk4 step
    state_hist[i+1] = rk4(state_hist[i], time_vector[i], dt, params, rotor_thrusts)
    
    # renormalize quaternion
    state_hist[i+1, 6:10] = state_hist[i+1, 6:10] / np.linalg.norm(state_hist[i+1, 6:10])

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
