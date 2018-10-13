from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G =  9.8 # acceleration due to gravity, in m/s^2
L1 = 1.0 # length of pendulum 1 in m
L2 = 1.0 # length of pendulum 2 in m
M1 = 1.0 # mass of pendulum 1 in kg
M2 = 1.0 # mass of pendulum 2 in kg


def deriv(u, t):

    du_dt = np.zeros_like(u)

    du_dt[0] = u[1]

    delt = u[2] - u[0]
    den1 = (M1+M2)*L1-M2*L1*cos(delt)*cos(delt)

    du_dt[1] = (M2*L1*u[1]*u[1]*sin(delt)*cos(delt)
               + M2*G*sin(u[2])*cos(delt) + M2*L2*u[3]*u[3]*sin(delt)
               - (M1+M2)*G*sin(u[0]))/den1

    du_dt[2] = u[3]

    den2 = (L2/L1)*den1
    du_dt[3] = (-M2*L2*u[3]*u[3]*sin(delt)*cos(delt)
               + (M1+M2)*G*sin(u[0])*cos(delt)
               - (M1+M2)*L1*u[1]*u[1]*sin(delt)
               - (M1+M2)*G*sin(u[2]))/den2

    return du_dt

dt = .025
t = np.linspace(0.0, 20, 1000)

th1 = 160.0
th2 = -30.0
w1 = 20.0
w2 = 5.0

u0 = np.array([th1, w1, th2, w2])*pi/180.

y = integrate.odeint(deriv, u0, t)

x1 = L1*sin(y[:,0])
y1 = -L1*cos(y[:,0])

x2 = L2*sin(y[:,2]) + x1
y2 = -L2*cos(y[:,2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([])
line3, = ax.plot([])
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], []) 
    line2.set_data([], [])
    line3.set_data([], [])
    time_text.set_text('')
    return line, line2, line3, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    plot2 = [x2[0:i],y2[0:i]]
    plot3 = [x1[0:i], y1[0:i]]
    line2.set_data(plot2)
    line3.set_data(plot3)
    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, line2, line3, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
    interval=5, blit=False, init_func=init)

#ani.save('C:/Users/Jacob/Desktop/double_pendulum.mp4', fps=60)

plt.show()