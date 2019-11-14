import numpy as np
import scipy.io
import scipy.sparse.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import random
import sys

num_iterations = 100
batch_size = 10

num_basis = 256
learning_rate = 0.01
sigma = 0.316
l = 1

image_width = 512
image_height = 512

basis_width = 16
basis_height = 16

def s(x):
    return -np.log(1.0 + x**2)

def s_prime(x):
    return (2*x) / (1.0 + x**2)

def sparseness(A):
    return -sum([s(a/sigma) for a in A])

def preserve_info(image, A, B):
    return -sum([(image[x][y] - sum([a * b[x][y]
        for a,b in zip(A,B)]))**2 for x in range(basis_width) for y in range(basis_height)])

def cost(A, image, B):
    return -preserve_info(image, A, B) - l * sparseness(B)

# coefficients (num_basis,) array
# basis_functions (num_basis, basis_width * basis_height) array
# image (basis_width * basis_height, ) array
def cost_prime(coefficients, basis_functions, image):
    reconstructed_image = np.zeros(image.shape)
    for i in range(num_basis):
        reconstructed_image += coefficients[i] * basis_functions[i]

    image_error = sum([i - b for (i,b) in zip(image, reconstructed_image)])

    sparsity = (l / sigma) * sum([s(a/sigma) for a in coefficients])

    return image_error + sparsity

# coefficients (num_basis,) array
# basis_functions (num_basis, basis_width * basis_height) array
# image (basis_width * basis_height, ) array
def cost_gradient(coefficients, basis_functions, image):
    gradient_vector = np.zeros(coefficients.shape)

    for i in range(num_basis):
        b_i = sum(np.multiply(basis_functions[i], image))
        C = 0
        for j in range(num_basis):
            C += coefficients[j] * sum(np.multiply(basis_functions[i], basis_functions[j]))
        D = -(l / sigma) * s_prime(coefficients[i] /sigma)
        gradient_vector[i] = learning_rate * (b_i - C - D)

    return gradient_vector

# coefficients (num_basis,) array
# basis_functions (num_basis, basis_width * basis_height) array
# image (basis_width * basis_height, ) array
def gradient_descent(coefficients, basis_functions, image):
    new_coefficients = coefficients.copy()
    gradient = np.ones(coefficients.shape)
    niters = 0

    cost = cost_prime(coefficients, basis_functions, image)

    while niters < 100:
        gradient = cost_gradient(new_coefficients, basis_functions, image)

        new_coefficients += learning_rate * gradient

        niters += 1

        print(str.format('descent iters: {}', niters))

        new_cost = cost_prime(coefficients, basis_functions, image)
        if (cost / new_cost) < 1e-2:
            return new_coefficients
        else:
            cost = new_cost

    return new_coefficients

if __name__ == "__main__":
    mat = scipy.io.loadmat('IMAGES.mat')
    mat = np.array(np.transpose(mat['IMAGES'], (2,0,1)))

    B = np.random.rand(num_basis, basis_height * basis_width)

    # iterate 1000 times
    for j in range(num_iterations):
        # choose a random image
        image = mat[random.randint(0, mat.shape[0] - 1)]

        subimages = {}

        I = np.zeros((batch_size, basis_height * basis_width))

        # choose 100 random sub images
        for i in range(batch_size):
            x = random.randint(0, image_width - basis_width - 1)
            y = random.randint(0, image_height - basis_height - 1)
            subimages[i] = image[x:x+basis_width,y:y+basis_height]
            I[i] = np.reshape(subimages[i], (basis_height * basis_width,))

        # calculate coefficients

        A = np.random.rand(batch_size, num_basis)

        for i in range(batch_size):
            res = gradient_descent(A[i], B, I[i])
            A[i] = res
            # res = optimize.minimize(cost_prime, A[i], (B, I[i]), method='CG', tol=0.01, options={'maxiter':100})
            # print(res.message)
            # print(str.format('iters: {}', res.nit))
            # print(res.x)
            # A[i] = res.x

        # calculate residual error

        R = I - (sum(A) / batch_size).dot(B)

        # update the base functions

        dB = np.zeros((num_basis, basis_width * basis_height))

        for i in range(batch_size):
            dB = dB + R[i].dot(A[i])

        dB = dB / batch_size

        B = B + learning_rate * dB

        print(str.format('iteration: {}/{}', j + 1, num_iterations))

    plt.imshow(B[0].reshape((basis_height, basis_width)))
    plt.show()