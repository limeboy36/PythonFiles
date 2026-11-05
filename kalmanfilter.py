import matplotlib.pyplot as plt
import numpy as np


Total_t = 50
r = 1 # guy at starting position
v = 3 # velocity of guy

sigmar = 0.25 # position covariance
sigmav = 0.25 # velocity covariance

x = np.array([[1],[3]])
x_len = x.shape[0]

R = np.array([[sigmar, 0],[0, sigmav]])

P = R

F = np.array([[1, 1], [0,1]])

H = np.eye(2)

Q = np.array([[0,0],[0,0]])

u = np.array([[0],[0]])

G = np.array([[0,0],[0,0]])

for Total_t in range(Total_t):

    r = r+3
    v = v

    # Should vary by sqrt(0.25) = 0.5
    pos_err = np.random.uniform(-1, 1) * 0.5
    vel_err = np.random.uniform(-1, 1) * 0.5
    r_est = r + pos_err
    v_est = v + vel_err
    z = np.array([[r_est],[v_est]])

    print(f"z value is: {z}")

    # Prediction Phase
    x = F @ x + G @ u

    print(f"x value is: {x}")

    P = F @ P @ F.T + Q

    # Update Phase
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    x = K @ z + (np.eye(x_len) - K) @ x

    P = (np.eye(x_len) - K @ H) @ P @ (np.eye(x_len) - K @ H).T + K @ R @ K.T

    print(f"updated x value is: {x}")
