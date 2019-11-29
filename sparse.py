import numpy as np
import scipy.io
import scipy.sparse.linalg as linalg
from scipy.optimize import minimize
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
import sys
import math
import time

num_iterations = 4000
batch_size = 100

num_basis = 64
learning_rate = 1e-3
ista_learning_rate = 1e-3

shrinkage_factor = 5e-3

image_width = 512
image_height = 512

basis_width = 8
basis_height = 8

runnum = 14

error_list = []

def cost(image, reconstructed, cf):
    reconstruction_error = sum(map(lambda x: x**2, (image - reconstructed)))
    sparse_error = sum(map(s, cf))
    return reconstruction_error + sparse_error

def s(x):
    return np.log(1 + x**2)

def normalize(X):
    return preprocessing.normalize(X, norm='l2')

def shrinkage_prime(X):
    return np.array(map(lambda z: map(lambda y: np.sign(y) * max(abs(y) - shrinkage_factor, 0), z), X))

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
    print(mat['IMAGES'].shape)
    mat = np.array(np.transpose(mat['IMAGES'], (2,0,1)))

    B = np.random.uniform(0, 1, (num_basis, basis_height * basis_width))
    B = normalize(B)

    for j in range(num_iterations):
        # choose a random image
        image = mat[random.randint(0, mat.shape[0] - 1)]

        subimages = np.zeros((batch_size, basis_height * basis_width))

        # choose 100 random sub images
        for i in range(batch_size):
            x = random.randint(0, image_width - basis_width - 1)
            y = random.randint(0, image_height - basis_height - 1)
            subimages[i] = np.reshape(image[x:x+basis_width, y:y+basis_height], (basis_height * basis_width,))

        # calculate coefficients
        coefficients = ista(subimages, B, 100)

        reconstructed_images = coefficients.dot(B)

        # calculate gradient
        gradient = -2.0 * (coefficients.transpose()).dot((subimages - reconstructed_images))

        print(gradient.mean())

        # update coefficients
        B = B - learning_rate * gradient

        normalize(B)

        total_cost = 0
        for i in range(batch_size):
            total_cost += cost(subimages[i], reconstructed_images[i], coefficients[i])

        error_list.append(total_cost / batch_size)

        print(total_cost / batch_size)

        print(str.format('iteration: {}/{}', j + 1, num_iterations))

        if (j + 1) % 10 == 0:
            graph_edge = int(math.sqrt(num_basis))
            fig, axes = plt.subplots(graph_edge, graph_edge)

            for r in range(num_basis):
                yindex = int(math.floor(r / graph_edge))
                xindex = int(r % graph_edge)

                axes[yindex, xindex].imshow(B[r].reshape((basis_height, basis_width)), cmap='gray')
                axes[yindex, xindex].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,
                    left=False,         # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False) # labels along the bottom edge are off

            fig.savefig(str.format('images/iteration_{}_run_{}', j + 1, runnum), dpi=400, bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

            fig1, axes2 = plt.subplots(2,1)
            axes2[0].imshow(np.reshape(subimages[0], (basis_width, basis_height)), cmap='gray')
            # axes2[0].set_title("Original")
            axes2[0].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,
                left=False,         # ticks along the top edge are off
                labelbottom=False,
                labelleft=False) # labels along the bottom edge are off
            axes2[1].imshow(np.reshape(reconstructed_images[0], (basis_height, basis_width)), cmap='gray')
            # axes2[1].set_title("Reconstructed image patch")
            axes2[1].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,
                left=False,         # ticks along the top edge are off
                labelbottom=False,
                labelleft=False) # labels along the bottom edge are off

            fig1.savefig(str.format("images/iteration_{}_run_{}_comparison", j + 1, runnum), dpi=400, bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig1)

            plt.clf()
            plt.plot(np.linspace(0,len(error_list), len(error_list)),error_list)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Average reconstruction cost over training")
            plt.savefig(str.format("images/iteration_{}_run_{}_cost", j + 1, runnum), dpi=400, bbox_inches = 'tight', pad_inches = 0)
            plt.close()