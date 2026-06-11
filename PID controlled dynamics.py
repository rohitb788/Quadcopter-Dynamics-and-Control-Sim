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

def quaternion_to_euler(q0, q1, q2, q3):
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    theta = np.arcsin(2*(q0*q2 - q3*q1))
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    
    return phi, theta, psi

def altitude_controller(z, z_desired, z_dot, dt, params, pid_gains, integral_error):
    error = z_desired-z
    de = -z_dot
    kp, ki, kd = pid_gains
    
    integral_error += error * dt
    
    p = kp * error
    i = ki * integral_error
    d = kd * de
    
    u_altitude = p + i + d
    
    #return new thrust and new integral error
    return ((params['m'] * params['g']) + u_altitude), integral_error

#output -> theta_desired
def x_controller(x, x_desired, x_dot, dt, pid_gains_x, integral_error_x):
    error = x_desired - x
    de = -x_dot
    kp, ki, kd = pid_gains_x
    
    integral_error_x += error * dt
    
    p = kp * error
    i = ki * integral_error_x
    d = kd * de
    
    theta_desired = - (p + i + d)
    
    theta_desired = np.clip(theta_desired, -0.3, 0.3)  # max ~17 degrees
    
    return theta_desired, integral_error_x

#output -> phi_desired
def y_controller(y, y_desired, y_dot, dt, pid_gains_y, integral_error_y):
    error = y_desired - y
    de = -y_dot
    kp, ki, kd = pid_gains_y
    
    integral_error_y += error * dt
    
    p = kp * error
    i = ki * integral_error_y
    d = kd * de
    
    phi_desired = p + i + d
    
    phi_desired = np.clip(phi_desired, -0.3, 0.3)  # max ~17 degrees
    
    return phi_desired, integral_error_y

def roll_controller(phi, phi_desired, wx, dt, pid_gains_roll, integral_error_roll):
    error = phi_desired - phi
    de = -wx
    kp, ki, kd = pid_gains_roll
    
    integral_error_roll += error * dt
    
    p = kp * error
    i = ki * integral_error_roll
    d = kd * de
    
    u_roll = p + i + d
    
    return u_roll, integral_error_roll

def pitch_controller(theta, theta_desired, wy, dt, pid_gains_pitch, integral_error_pitch):
    error = theta_desired - theta
    de = -wy
    kp, ki, kd = pid_gains_pitch
    
    integral_error_pitch += error * dt
    
    p = kp * error
    i = ki * integral_error_pitch
    d = kd * de
    
    u_pitch = p + i + d
    
    return u_pitch, integral_error_pitch

def yaw_controller(psi, psi_desired, wz, dt, pid_gains_yaw, integral_error_yaw):
    error = psi_desired - psi
    de = -wz
    kp, ki, kd = pid_gains_yaw
    
    integral_error_yaw += error * dt
    
    p = kp * error
    i = ki * integral_error_yaw
    d = kd * de
    
    u_yaw = p + i + d
    
    return u_yaw, integral_error_yaw

def mixer(T_total, u_roll, u_pitch, u_yaw, params):
    l = params['l']
    kd = params['kd']
    M = np.array([[1, 1, 1, 1], 
                  [-l, -l, l, l],
                  [-l, l, l, -l],
                  [kd, -kd, kd, -kd]])
    
    M1 = np.linalg.inv(M)
    
    u_vector = np.array([T_total, u_roll, u_pitch, u_yaw])
    
    return M1 @ u_vector

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
wx = 0
wy = 0
wz = 0

state = np.array([x_pos, y_pos, z_pos, x_dot_pos, y_dot_pos, z_dot_pos, q0, q1, q2, q3, wx, wy, wz])

rotor_thrusts = [2.2525, 2.2525, 2.6525, 2.6525]

z_desired = 5.0        # m
x_desired = 3.0
y_desired = 2.0
phi_desired = 0.0      # rad
theta_desired = 0.0    # rad
psi_desired = 0.0      # rad

pid_altitude = [5.0, 0.1, 3.0]   # kp, ki, kd
pid_roll =  [1.0, 0.0, 0.5]
pid_pitch = [1.0, 0.0, 0.5]
pid_yaw =      [2.0, 0.0, 1.0]
pid_x = [0.1, 0.0, 0.5]
pid_y = [0.1, 0.0, 0.5]

integral_errors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # altitude, roll, pitch, yaw, x, y

#time setting of simuation
t_start = 0 
t_end = 10
dt = 0.01
time_vector = np.arange(t_start, t_end + dt, dt)
state_hist = np.zeros((len(time_vector), 13))

#simulation loop
state_hist[0] = state
for i in range(len(time_vector) - 1):
    # 1. extract current state variables
    x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, wx, wy, wz = state_hist[i]
    
    # 2. convert quaternion to euler angles
    phi, theta, psi = quaternion_to_euler(q0, q1, q2, q3)
    
    # 3. run controllers
    theta_desired, integral_errors[4] = x_controller(x, x_desired, x_dot, dt, pid_x, integral_errors[4])
    phi_desired, integral_errors[5] = y_controller(y, y_desired, y_dot, dt, pid_y, integral_errors[5])
    
    T_total, integral_errors[0] = altitude_controller(z, z_desired, z_dot, dt, params, pid_altitude, integral_errors[0])
    u_roll, integral_errors[1] = roll_controller(phi, phi_desired, wx, dt, pid_roll, integral_errors[1])
    u_pitch, integral_errors[2] = pitch_controller(theta, theta_desired, wy, dt, pid_pitch, integral_errors[2])
    u_yaw, integral_errors[3] = yaw_controller(psi, psi_desired, wz, dt, pid_yaw, integral_errors[3])
    
    # 4. run mixer
    rotor_thrusts = mixer(T_total, u_roll, u_pitch, u_yaw, params)
    
    # 5. rk4 step
    state_hist[i+1] = rk4(state_hist[i], time_vector[i], dt, params, rotor_thrusts)
    
    # 6. renormalize quaternion
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
axes[0].axhline(y=z_desired, color='r', linestyle='--')
axes[0].set_ylabel('Z (m)')
axes[1].plot(time_vector, state_hist[:, 0])
axes[1].axhline(y=x_desired, color='r', linestyle='--')
axes[1].set_ylabel('X (m)')
axes[2].plot(time_vector, state_hist[:, 1])
axes[2].axhline(y=y_desired, color='r', linestyle='--')
axes[2].set_ylabel('Y (m)')
axes[2].set_xlabel('Time (s)')
plt.suptitle('Position vs Time')
plt.tight_layout()
plt.show()