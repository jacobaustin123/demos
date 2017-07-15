import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

fig, ax = plt.subplots()

plot = ax.scatter([], [])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

n=10000

x = np.zeros(n)

for i in range(n):
    sample = np.random.beta(2, 5, 2000)
    x[i] = np.mean(sample)

mu = np.mean(x)
std = np.std(x)

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, std)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Average of 2000 Samples')
plt.ylabel('Frequency')
plt.title(r'$\mathrm{Central\ Limit\ Theorem\ Example}$')
plt.axis([mu-5*std, mu+5*std, 0, 1.15*np.max(y)])
plt.grid(True)

plt.show()