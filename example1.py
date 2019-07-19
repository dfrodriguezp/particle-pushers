"""
Leapfrog scheme for simple harmonic oscillator
"""

import numpy
from matplotlib import pyplot
import matplotlib.animation as animation

PI = numpy.pi
dt = 0.1
ang_freq = 1.0
T = 2 * PI / ang_freq
steps = int(4 * T / dt) + 1
amplitude = 0.1
anim = True


def update_velocity(pos, vel, dt):
    acc = -(ang_freq ** 2) * pos
    vel_new = vel + acc * dt
    return vel_new


def leapfrog(pos, vel, dt):
    vel_new = update_velocity(pos, vel, dt)
    pos_new = pos + vel_new * dt
    return pos_new, vel_new


def update_lines(i, lines, data_lines):
    for line, data in zip(lines, data_lines):
        line.set_data(data[0:2, :i])


def create_lines(axis, data, labels, linestyles):
    return [axis.plot(dat[0, 0:1], dat[1, 0:1], label=label, ls=ls)[0] for dat, label, ls in zip(data, labels, linestyles)]


def main():
    t = numpy.linspace(0, steps * dt, steps)
    x_th = amplitude * numpy.cos(ang_freq * t)
    v_th = -amplitude * ang_freq * numpy.sin(ang_freq * t)
    x_num = numpy.zeros(steps)
    v_half = numpy.zeros(steps)
    v_num = numpy.zeros(steps)
    x_num[0] = amplitude
    v_half[0] = update_velocity(x_num[0], v_half[0], -0.5 * dt)

    for i in range(steps-1):
        x_num[i+1], v_half[i+1] = leapfrog(x_num[i], v_half[i], dt)
        v_num[i+1] = update_velocity(x_num[i+1], v_half[i+1], 0.5 * dt)

    # Creating lines data for axis 1
    data1_th = numpy.stack((t, x_th))
    data1_num = numpy.stack((t, x_num))
    data1 = [data1_th, data1_num]

    # Creating lines data for axis 2
    data2_th = numpy.stack((x_th, v_th))
    data2_num = numpy.stack((x_num, v_num))
    data2 = [data2_th, data2_num]

    labels = ["Theory", "Numerical"]
    linestyles = ["-", "--"]

    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(14, 9))
    fig.suptitle("Simple harmonic oscillator (Leapfrog)", fontsize=20)

    ax1.set_xlim(0, steps * dt)
    ax1.set_ylim(-3 * amplitude, 3 * amplitude)
    ax1.set_xlabel("Time $[s]$", fontsize=15)
    ax1.set_ylabel("Position $[m]$", fontsize=15)
    ax1.grid(ls="--")

    ax2.set_xlim(-3 * amplitude, 3 * amplitude)
    ax2.set_ylim(-3 * amplitude, 3 * amplitude)
    ax2.set_xlabel("Position $[m]$", fontsize=15)
    ax2.set_ylabel("Velocity $[m/s]$", fontsize=15)
    ax2.grid(ls="--")

    if anim:
        lines1 = create_lines(ax1, data1, labels, linestyles)
        lines2 = create_lines(ax2, data2, labels, linestyles)
        point1 = ax1.scatter([], [], c="darkblue")
        point2 = ax2.scatter([], [], c="darkblue")
        ax1.legend(loc="upper left", fontsize=15)
        ax2.legend(loc="upper left", fontsize=15)

        def animate(i):
            update_lines(i, lines1, data1)
            update_lines(i, lines2, data2)
            x1, y1 = data1[0][0:2, i]
            x2, y2 = data2[0][0:2, i]
            point1.set_offsets((x1, y1))
            point2.set_offsets((x2, y2))
            return lines1, lines2, point1, point2

        plot_anim = animation.FuncAnimation(
            fig, animate, frames=steps, interval=10)
        pyplot.show()

    else:
        ax1.plot(t, x_th, label="Theory")
        ax1.plot(t, x_num, ls="--", label="Numerical")

        ax2.plot(x_th, v_th, label="Theory")
        ax2.plot(x_num, v_num, ls="--", label="Numerical")

        ax1.legend(loc="best")
        ax2.legend(loc="best")
        pyplot.savefig("sho_leapfrog.pdf")
        pyplot.close()


if __name__ == "__main__":
    main()
