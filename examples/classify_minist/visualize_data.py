import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from download_mnist import load_mnist

X, y, X_test, y_test = load_mnist()

print(X[10, :])

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

imgs = []
for i in range(10):
	idx = np.random.choice(len(X))
	img = X[idx, :]
	imgs.append(img)
for ax, im in zip(grid, imgs):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

plt.show()
