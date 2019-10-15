import matplotlib.pyplot as plt
import numpy as np
from random import uniform

ms = 0.001
mV = 0.001
MOhm = 1000000
nA = 0.000000001
tau_m = 10*ms
E_l = -70*mV
V_t = -40*mV
R_m = 10*MOhm
I_e = 3.1*nA
dt = 0.001
V_r = E_l

times = np.arange(0, 1, dt)
# times = np.arange(0, 0.4, dt)
Vs = [V_r]

def differential(V):
    dV = (E_l - V + R_m*I_e)/tau_m
    return dV

for i in times:
    prev = Vs[-1]
    next = prev + differential(prev)*ms
    if (next >= V_t):
        next = V_r

    Vs.append(next)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.plot(times, Vs[:len(times)], color = "black", label = "voltage")
ax.axhline(-40*mV, linestyle = "--", alpha = 0.7, label = "threshold", color = "crimson")
ax.axhline(-70*mV, linestyle = "--", alpha = 0.7, label = "leak potential")
ax.legend()
plt.ylim(-75*mV, -20*mV)
plt.xlabel("Time (s)", fontsize = 14)
plt.ylabel("Voltage (mV)", fontsize = 14)
ax.tick_params(labelsize=14)
plt.show()

fig.savefig("Int_Fire_1second")

tau_m = 20*ms
E_l = -70*mV
V_t = -54*mV
# R_m = 10*MOhm
# I_e = 3.1*nA
dt = 0.001
V_r = -80*mV
R_mI_e = 18*mV
Rmgs = 0.15
P = 0.5
ax.legend()
tau_s = 10*ms
s_1 = 0
s_2 = 0
E_excite = 0
E_inhibit = -80*mV

V1s = [uniform(V_r, V_t)]
V2s = [uniform(V_r, V_t)]
times = np.arange(0, 1, dt)

# Need a function for calculating s
def dS(s):
    return -s/tau_s


def neuron(V, s, E_s):
    dV = (E_l - V + R_mI_e + Rmgs*s*(E_s-V))/tau_m
    return dV

for t in times:
    prev_1 = V1s[-1]
    prev_2 = V2s[-1]

#     next_1 = prev_1 + neuron(prev_1, s_2, E_excite)*dt
#     next_2 = prev_2 + neuron(prev_2, s_1, E_excite)*dt

    next_1 = prev_1 + neuron(prev_1, s_2, E_inhibit)*dt
    next_2 = prev_2 + neuron(prev_2, s_1, E_inhibit)*dt

    s_1 = s_1 + dS(s_1) * dt
    s_2 = s_2 + dS(s_2) * dt

    if (next_1 >= V_t):
        next_1 = V_r
        s_1 += P
    V1s.append(next_1)

    if (next_2 >= V_t):
        next_2 = V_r
        s_2 += P
    V2s.append(next_2)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.plot(times, V1s[:len(times)], color = "crimson", label = "Neuron 1 Voltage")
ax.plot(times, V2s[:len(times)], color = "mediumblue", label = "Neuron 2 Voltage")
ax.axhline(-54*mV, linestyle = "-.", alpha = 0.7, label = "Firing Threshold", color = "black")
ax.axhline(-80*mV, linestyle = "--", alpha = 0.7, label = "Leak Potential", color = "black")
ax.legend()
ax.tick_params(labelsize=14)
plt.xlabel("Time (seconds)", fontsize = 14)
plt.ylabel("Voltage (mV)", fontsize = 14)
plt.ylim(-85*mV, -30*mV)
# fig.suptitle('Excitatory Synapse', fontsize=16)
plt.show()
fig.savefig("Int_Fire_Excitatory")

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.plot(times, V1s[:len(times)], color = "crimson", label = "Neuron 1 Voltage")
ax.plot(times, V2s[:len(times)], color = "mediumblue", label = "Neuron 2 Voltage")
ax.axhline(-54*mV, linestyle = "-.", alpha = 0.7, label = " Firing Threshold", color = "black")
ax.axhline(-80*mV, linestyle = "--", alpha = 0.7, label = "Leak Potential", color = "black")
ax.legend()
ax.tick_params(labelsize=14)
plt.xlabel("Time (seconds)", fontsize = 14)
plt.ylabel("Voltage (mV)", fontsize = 14)
plt.ylim(-85*mV, -30*mV)
plt.show()
fig.savefig("Int_Fire_Inhibitory")
