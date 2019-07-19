"""
Motion of a charged particle in a uniform magnetic field
"""

from matplotlib import pyplot
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy

PI = numpy.pi
TWO_PI = 2 * PI


def projection(A, B):
    paral = numpy.dot(A, B) * B / numpy.linalg.norm(B) ** 2
    perp = A - paral
    return paral, perp


def update_lines(i, lines, data_lines):
    for line, data in zip(lines, data_lines):
        line.set_data(data[0:2, :i])
        line.set_3d_properties(data[2, :i])


def update_velocity(vel, dt):
    global charge, mass, E, B
    vel_new = vel + (charge / mass) * (E + numpy.cross(vel, B)) * dt
    return vel_new


def boris(vel, dt):
    global charge, mass, E, B

    u = (charge / mass) * B * 0.5 * dt
    s = (2 * u) / (1 + numpy.linalg.norm(u) ** 2)

    v_minus = vel + (charge * E / mass) * 0.5 * dt
    v_prime = v_minus + numpy.cross(v_minus, u)
    v_plus = v_minus + numpy.cross(v_prime, s)
    vel_new = v_plus + (charge * E / mass) * 0.5 * dt
    return vel_new


def leapfrog(pos, vel, dt):
    vel_new = boris(vel, dt)
    pos_new = pos + vel_new * dt
    return pos_new, vel_new


# Definition of parameters
B = numpy.array([0.0, 0.0, 1.0])
E = numpy.array([0.1, 0.1, 0.0])
vel = numpy.array([1.0, 0.0, 1.0])
mass = 1.0
charge = -1.0

dt = 0.1
x0, y0, z0 = 0, 0, 0

v_paral, v_perp = numpy.linalg.norm(projection(vel, B), axis=1)
omega = abs(charge) * numpy.linalg.norm(B) / mass
r_L = v_perp / omega
T = TWO_PI / omega
steps = int(5 * T / dt) + 1

t = numpy.linspace(0, steps * dt, steps)

# Analytic solution for E = [Ex, Ey, 0] and B = [0, 0, Bz]
x_th = x0 + (E[1] / B[2]) * t - (abs(charge) / charge) * \
    r_L * numpy.cos(omega * t)
y_th = y0 - (E[0] / B[2]) * t + r_L * numpy.sin(omega * t)
z_th = z0 + v_paral * t

pos_num = numpy.zeros((steps, 3))
vel_half = numpy.zeros((steps, 3))
vel_num = numpy.zeros((steps, 3))

pos_num[0] = x0 - (abs(charge) / charge) * r_L, y0, z0
vel_num[0] = E[1] / B[2], v_perp - E[0] / B[2], v_paral

vel_half[0] = boris(vel_num[0], -0.5*dt)
for i in range(steps-1):
    pos_num[i+1], vel_half[i+1] = leapfrog(pos_num[i], vel_half[i], dt)
    vel_num[i+1] = boris(vel_half[i+1], 0.5*dt)

pos_th = numpy.stack((x_th, y_th, z_th))
data = [pos_th, pos_num.T]

fig = pyplot.figure()
ax = fig.add_subplot(111, projection="3d")
labels = ["Theory", "Numerical"]
linestyles = ["-", "--"]

lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], label=label, ls=ls)[
    0] for dat, label, ls in zip(data, labels, linestyles)]
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
ax.set_zlabel("$z$", fontsize=20)

ax.set_xlim(min(data[0][0, :]), max(data[0][0, :]))
ax.set_ylim(min(data[0][1, :]), max(data[0][1, :]))
ax.set_zlim(min(data[0][2, :]), max(data[0][2, :]))
ax.legend(loc="best")


def animate(i):
    update_lines(i, lines, data)
    return lines


line_anim = animation.FuncAnimation(
    fig, animate, frames=steps, interval=25)
pyplot.show()
