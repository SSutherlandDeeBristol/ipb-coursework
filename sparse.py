import numpy as np
import scipy.io
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mat = scipy.io.loadmat('IMAGES.mat')
    mat = np.transpose(mat['IMAGES'], (2,0,1))

    for i in range(10):
        plt.imshow(mat[i], cmap='gray')
        plt.show()