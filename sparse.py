import numpy as np
import scipy.io
import scipy.sparse.linalg as linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
import sys
import math

num_iterations = 4000
batch_size = 100

num_basis = 64
learning_rate = 5e-2
ista_learning_rate = 1e-2

shrinkage_factor = 5e-3

image_width = 512
image_height = 512

basis_width = 8
basis_height = 8

def normalize(X):
    return (X / math.sqrt(sum([y**2 for y in X.reshape(num_basis * basis_width * basis_height,)])))

def shrinkage(X):
    return (np.array(map(lambda z: map(lambda y: max(y, 0), z), (X - shrinkage_factor)))
          - np.array(map(lambda z: map(lambda y: max(y, 0), z), (-1 * X - shrinkage_factor))))

# basis_functions (num_basis, basis_width * basis_height) array
# image (batch_size, basis_width * basis_height) array
def ista(images, basis_functions, E):
    weights = np.random.uniform(0, 1, (batch_size, num_basis))

    for e in range(E):
        # calculate gradient
        reconstructed_images = weights.dot(basis_functions)
        gradient = -2.0 * ((images - reconstructed_images).dot(basis_functions.transpose()))
        # adjust weights
        weights = weights - ista_learning_rate * gradient
        # ensure sparsity
        weights = shrinkage(weights)

    return weights

if __name__ == "__main__":
    mat = scipy.io.loadmat('IMAGES.mat')
    mat = np.array(np.transpose(mat['IMAGES'], (2,0,1)))

    #mat -= mat.min()
    #mat *= 1/mat.max()

    B = np.random.rand(num_basis, basis_height * basis_width)
    B = normalize(B)

    # iterate 1000 times
    for j in range(num_iterations):
        # choose a random image
        image = mat[random.randint(0, mat.shape[0] - 1)]

        subimages = np.zeros((batch_size, basis_height * basis_width))

        # choose 100 random sub images
        for i in range(batch_size):
            x = random.randint(0, image_width - basis_width - 1)
            y = random.randint(0, image_height - basis_height - 1)
            subimages[i] = np.reshape(image[x:x+basis_width,y:y+basis_height], (basis_height * basis_width,))

        # calculate coefficients
        R = ista(subimages, B, 300)

        reconstructed_images = R.dot(B)

        gradient = -2.0 * (R.transpose()).dot((subimages - reconstructed_images))

        # update coefficients
        B = B - learning_rate * gradient

        normalize(B)

        print(str.format('iteration: {}/{}', j + 1, num_iterations))

        if (j + 1) % 10 == 0:
            graph_edge = int(math.sqrt(num_basis))
            fig, axes = plt.subplots(graph_edge, graph_edge)

            for r in range(num_basis):
                yindex = int(math.floor(r / graph_edge))
                xindex = int(r % graph_edge)

                axes[yindex, xindex].imshow(B[r].reshape((basis_height, basis_width)), cmap='gray')

            fig.savefig(str.format('images/iteration_{}', j + 1), dpi=400)