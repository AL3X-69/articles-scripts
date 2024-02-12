# This script is made by Colejhudson and fixed by me (Alex6), originally found on
# https://commons.wikimedia.org/wiki/File:Vector_Field_of_a_Function%27s_Gradient_imposed_over_a_Color_Plot_of_that_Function.svg

import matplotlib.pyplot as plt
import numpy as np


# Returns the partial derivative with respect to the `i`th parameter of the
# given function. This is done using the method of finite differences
def partial_derivative(f, i=0, h=0.001e-10):
    def partial(*xs):
        dxs = list(xs)
        dxs[i] = xs[i] + h
        return (f(*dxs) - f(*xs)) / h

    return partial


# Returns an np.array (a vector) of partial derivatives with respect to each
# parameter given, ie the gradient
def gradient(f):
    def gradient_of_f(*xs):
        return np.array([partial_derivative(f, i)(*xs) for i in range(len(xs))])

    return gradient_of_f


# The function to be plotted. It contains, by construction, an adjacent 'sink' and
# 'source'. These features demonstrate that the gradient operator yields a vector
# field wherein each vector points from sinks (lower values) to sources (higher values)
def f(x, y):
    return x * np.exp(-1 * (x ** 2 + y ** 2))


# Here we construct our domain (the inputs) by discretizing the space between -2 and 2
# into 25 evenly spaced points. This is done for both the x and y dimensions.
n = 25
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)

# Then we generate a grid of points from our two dimensions, on which we'll evaluate
# the above function
x, y = np.meshgrid(x, y)

# Then we do said evaluating
z = f(x, y)

# And generate it's gradient. Note that `gradient` returns a single array, for plotting
# convenience I've separated out the the dx and dy parts.
dzdx, dzdy = gradient(f)(x, y)

# And finally we generate the plot
plt.figure(figsize=(8, 4))
plt.pcolormesh(x, y, z)
plt.quiver(x, y, dzdx, dzdy, scale=n, pivot='mid', headwidth=4, minshaft=2, minlength=2)
plt.colorbar()
plt.savefig("vector-field.png")
plt.show()
