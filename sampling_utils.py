###################################################################
# Low-discrepency sampling
###################################################################
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
import numpy as np

def sampling_sphere_surface(n_samples, n_dim):
    # dims = []
    # for d in range(n_dim - 1):
    #     dims.append((0., 1.))
    # space = Space(dims)
    # sampler = Hammersly()
    # x = sampler.generate(space.dimensions, n_samples)
    # x = np.array(x)
    # for n in range(n_dim - 1):
    #     if n == 0:
    #         x[:, n] = x[:, n] * 2 * np.pi
    #     else:
    #         x[:, n] = 1 - 2 * x[:, n]
    # x_s = []
    # for n in range(n_dim - 1):
    #     if n == 0:
    #         x_s = np.column_stack([np.cos(x[:, n:n + 1]), np.sin(x[:, n:n + 1])])
    #     else:
    #         x_s = np.column_stack([np.sqrt(1 - x[:, n:n + 1] ** 2) * x_s, x[:, n:n + 1]])
    # # print(x_s)
    # fig = plt.figure(figsize=[5,5])
    # plt.scatter(x_s[:,0], x_s[:,1])
    # plt.show()
    x = sampling_sphere(n_samples, n_dim)
    x_s = x/np.linalg.norm(x,axis=1,ord=2, keepdims=True)
    return x_s


def sampling_sphere(n_samples, n_dim):
    dims = []
    for d in range(n_dim):
        dims.append((0., 1.))
    space = Space(dims)
    sampler = Hammersly()
    x = sampler.generate(space.dimensions, n_samples)
    x = np.array(x)

    r = (x[:, [0]]) ** (1 / n_dim)
    # print(r)
    # print(1/0)
    for n in range(1, n_dim):
        if n == 1:
            x[:, n] = x[:, n] * 2 * np.pi
        else:
            x[:, n] = 1 - 2 * x[:, n]
    x_s = []
    for n in range(1, n_dim):
        if n == 1:
            x_s = np.column_stack([np.cos(x[:, n:n + 1]), np.sin(x[:, n:n + 1])])
        else:
            x_s = np.column_stack([np.sqrt(1 - x[:, n:n + 1] ** 2) * x_s, x[:, n:n + 1]])
    return np.multiply(x_s, r)


def sampling_square(n_samples, n_dim):
    dims = []
    for d in range(n_dim):
        dims.append((-1., 1.))
    space = Space(dims)
    sampler = Hammersly()
    x = sampler.generate(space.dimensions, n_samples)
    # x = np.random.uniform(-1,1,[n_samples,n_dim])
    x = np.array(x)
    return x


def sampling_square_surface(n_samples, n_dim):
    # num_face = 2 * n_dim
    # dims = []
    # for d in range(n_dim - 1):
    #     dims.append((-1., 1.))
    # space = Space(dims)
    # sampler = Hammersly()
    # x = []
    # for i in range(2 * n_dim):
    #     if i % 2 == 0:
    #         xt = np.ones([n_samples // num_face, n_dim]) * 1
    #     else:
    #         xt = np.ones([n_samples // num_face, n_dim]) * -1
    #     xs = sampler.generate(space.dimensions, n_samples // num_face)
    #     index = list(set(range(n_dim)) - set([i // 2]))
    #     xt[:, index] = np.array(xs)
    #     x.append(xt)
    # x = np.concatenate(x, axis=0)
    x = sampling_square(n_samples, n_dim)
    x_s = x/np.linalg.norm(x,axis=1,ord=np.inf, keepdims=True)
    return x_s


def sampling_body(n_samples, n_dim, shape, lu=[-1,1]):
    if shape == 'sphere':
        x = sampling_sphere(n_samples, n_dim)
        return x + (lu[0]+lu[1])/2
    elif shape == 'square':
        x = sampling_square(n_samples, n_dim)
        return (x+1)*(lu[1]-lu[0])/2 + lu[0] #x + (lu[0]+lu[1])/2#


def sampling_surface(n_samples, n_dim, shape, lu=[-1,1]):
    if shape == 'sphere':
        x = sampling_sphere_surface(n_samples, n_dim)
        return x + (lu[0]+lu[1])/2
    elif shape == 'square':
        x = sampling_square_surface(n_samples, n_dim)
        return (x+1)*(lu[1]-lu[0])/2 + lu[0] #x + (lu[0]+lu[1])/2#
