import numpy as np
import matplotlib.pyplot as plt

size = 400
num_iter = 25

image = np.zeros(shape=(size, size))

for x in range(size):
    for y in range(size):
        zx, zy = cx, cy = -2 + 2.5 * x / size, -1.25 + 2.5 * y / size
        for i in range(num_iter):
            zx, zy = zx**2 - zy**2 + cx, 2*zx*zy + cy
            if zx**2 + zy**2 > 4:
                break

        image[y, x] = 255 - 255*(i/(num_iter - 1))

plt.imshow(image, cmap='gray')
plt.show()