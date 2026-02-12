import torch
import numpy as np


l = 0.5
k1 = 0.000000045
T_cmd = 5.0
tau_cmd = 0.0

T_and_tau = np.array([[T_cmd], [tau_cmd]]) 
# T_and_tau = np.array([T_cmd, tau_cmd]) 
# print(T_and_tau)

alloc_mat = np.array([[1.0, 1.0], [-l, l]])
# print(np.linalg.inv(alloc_mat) @ T_and_tau / k1)

rotor_speeds = np.sqrt((np.linalg.inv(alloc_mat) / k1) @ T_and_tau)
print(rotor_speeds)