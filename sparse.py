import numpy as np
import scipy.io
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mat = scipy.io.loadmat('IMAGES.mat')
    mat = mat['IMAGES']

    images = {}

    for i in range(10):
        images[i] = np.zeros((512,512))

    for x in range(512):
        for y in range(512):
            for z in range(10):
                (images[z])[x][y] = mat[x][y][z]

    for i in range(10):
        plt.imshow(images[i], cmap='gray')
        plt.show()