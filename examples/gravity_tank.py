import matplotlib.pyplot as plt
from models.gravity_tank import GravityTankParams, f, steady_state
from core.integrators import euler_explicit

from dataclasses import replace

p_100 = GravityTankParams()
p_50 = GravityTankParams(F0=0.5 * p_100.F0)
p_67 = GravityTankParams(F0=0.67 * p_100.F0)

y_50 = steady_state(p_50)
y_67 = steady_state(p_67)
y_100 = steady_state(p_100)

p_A_100 = GravityTankParams(At=10 * p_100.At)
p_A_50 = GravityTankParams(F0=0.5 * p_A_100.F0)
p_A_67 = GravityTankParams(F0=0.67 * p_A_100.F0)

y_A_50 = steady_state(p_A_50)
y_A_67 = steady_state(p_A_67)
y_A_100 = steady_state(p_A_100)

def format_ax(ax):
    ax.axhline(7.0, color="k", ls="--", label="tank top 7 ft")
    ax.set_xlabel("t, s")
    ax.set_ylabel("h, ft")
    ax.legend()
    ax.grid(True)

def plot_initial_states(ax):
    for y0 in (y_50, y_67, y_100):
        t, Y = euler_explicit(f, y0, (0.0, 600.0), 1.0, args=(p_100,))
        ax.plot(t, Y[:, 1], label=f"h0 = {y0[1]:.2f} ft³/s")
    format_ax(ax)

def plot_feed_steps_upward(ax):
    for F0 in (p_100.F0, 1.1 * p_100.F0, 1.2 * p_100.F0):
        t, Y = euler_explicit(f, y_67, (0.0, 600.0), 1.0, args=(replace(p_100, F0=F0),))
        ax.plot(t, Y[:, 1], label=f"F0 = {F0:.2f} ft³/s")
    format_ax(ax)

def plot_feed_steps_downward(ax):
    for F0 in (p_100.F0, 0.9 * p_100.F0, 0.8 * p_100.F0, 0.5 * p_100.F0):
        t, Y = euler_explicit(f, y_100, (0.0, 1000.0), 1.0, args=(replace(p_100, F0=F0),))
        ax.plot(t, Y[:, 1], label=f"F0 = {F0:.2f} ft³/s")
    format_ax(ax)

def plot_tank_area_increase(ax):
    for y0 in (y_A_50, y_A_67, y_A_100):
        t, Y = euler_explicit(f, y0, (0.0, 600.0), 1.0, args=(p_A_100,))
        ax.plot(t, Y[:, 1], label=f"h0 = {y0[1]:.2f} ft³/s")
    format_ax(ax)

fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(1, 4, sharey=True, figsize=(20, 6))
plot_initial_states(ax_a)
plot_feed_steps_upward(ax_b)
plot_feed_steps_downward(ax_c)
plot_tank_area_increase(ax_d)
plt.show()