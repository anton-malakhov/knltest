import timeit
import numpy as np
from numpy.random import random as rnd
from numpy.polynomial.legendre import legval as np_legval
import numba as nb


x_size = (100,10000)
c_size = 250
iterations = 1

def np_kernel(x, c):
    nd = len(c)
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        nd = nd - 1
        c0 = c[-i] - c1*(1 - 1/nd)
        c1 = tmp + (c1*(2 - 1/nd))*x
    return c0 + c1*x

def nb_kernel(x, c, r):
    for xi in range(x.size):
      nd = len(c)
      c0 = c[-2]
      c1 = c[-1]
      for ci in range(3, len(c) + 1):
        tmp = c0
        nd = nd - 1
        c0 = c[-ci] - c1*(1 - 1/nd)
        c1 = tmp + (c1*(2 - 1/nd))*x[xi]
      r[xi] = c0 + c1*x[xi]
nbj_kernel = nb.njit('void(f8[:],f8[:],f8[:])', fastmath=True)(nb_kernel)
nbg_kernel = nb.guvectorize('f8[::1],f8[::1],f8[::1]', '(n),(m)->(n)', nopython=True, target="parallel", fastmath=True)(nb_kernel) # , target="parallel"

def legval_with(ufunc, x, c, args=(), tensor=True):
    c = np.array(c, ndmin=1, copy=0)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
#    if isinstance(x, np.ndarray) and tensor:
#        c = c.reshape(c.shape + (1,)*x.ndim)   # I don't know how to make it work with Numba

    if len(c) == 1:
        c0 = c[0]
        return c0 + x
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
        return c0 + c1*x # not optimal, merge with below?
    else:
        return ufunc(x, c, *args)


class bench:
    def setup(self):
        self.c = rnd(c_size)
        self.x = rnd(x_size)

    def time_np(self):
        return np_legval(self.x, self.c)

    def time_npm(self):
        return legval_with(np_kernel, self.x, self.c)

    def time_nbg(self):
        return legval_with(nbg_kernel, self.x, self.c)

    def time_nbj(self):
        r = np.empty(self.x.shape)
        print("Shape:", self.c.shape)
        legval_with(nbj_kernel, self.x, self.c, (r,))
        return r

def check(orig, probe):
    print("Shapes:", orig.shape, probe.shape)
    if not np.allclose(orig, probe):
       if orig.shape != probe.shape:
          print("Mismatch shapes:", orig.shape, probe.shape)
       else:
          print("Original:", orig)
          print("Mismatch:", probe)
       assert False, "Mismath"

if __name__ == "__main__":
    b = bench()
    b.setup()
    r = b.time_np()
    check(r, b.time_nbg())
#    check(r, b.time_nbj())
    check(r, b.time_npm())
    setup = 'from __main__ import bench as B; b=B();b.setup()'
#    print("Numba.jit : ", timeit.repeat('b.time_nbj()', setup, number=iterations, repeat=3))
    print("GUfunc    : ", timeit.repeat('b.time_nbg()', setup, number=iterations, repeat=3))
    print("Mod  Numpy: ", timeit.repeat('b.time_npm()', setup, number=iterations, repeat=3))
    print("Orig Numpy: ", timeit.repeat('b.time_np()', setup, number=iterations, repeat=3))
