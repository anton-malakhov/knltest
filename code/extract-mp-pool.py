#!/usr/bin/env python

"""
Test of DESI spectral extraction code with multiprocessing parallelism

batchopts="-C haswell -p debug -t 00:20:00"
srun -n 1 -c 64 $batchopts python extract-mp.py 1 4 16 32 64 --bundlesize 5 --numwave 50
"""

from __future__ import absolute_import, division, print_function
import time
t0 = time.time()

import sys, os
import platform
import optparse
import multiprocessing as mp
import time

import numpy as np

from specter.extract import ex2d
import specter.psf
import knltest

t1 = time.time()
print('wakeup time {:.1f}'.format(t1-t0))

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-p", "--psf", type=str,  help="input psf file")
# parser.add_option("-n", "--numspec", type=int, default=100, help="number of spectra")
parser.add_option("-w", "--numwave", type=int, default=200, help="number of wavelengths")
parser.add_option("-b", "--bundlesize", type=int, default=25, help="size of bundles of spectra")

opts, ntest = parser.parse_args()

#- OMP environment
# os.environ['OMP_PROC_BIND']='spread'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['KNL_NUM_THREADS'] = '1'
# os.environ['OMP_PLACES'] = 'cores("1")'

#- OMP_NUM_THREADS options to test
if len(ntest) == 0:
    ntest = (1,4,16,32,64)

#- Load point spread function model
if opts.psf is None:
    thisdir = os.path.split(knltest.__file__)[0]
    opts.psf = thisdir + '/../etc/psfnight-r0.fits'
    assert os.path.exists(opts.psf)


psf = specter.psf.load_psf(opts.psf)

#- Create fake noisy image
ny, nx = psf.npix_y, psf.npix_x
image = np.random.normal(loc=0, scale=1, size=(ny,nx))
imageivar = np.ones_like(image)

#- Spectra and wavelengths to extract
w = np.linspace(psf.wmin_all, psf.wmax_all, 2000)

#- Wake up the code in case there is library loading overhead
flux, ivar, R = ex2d(image, imageivar, psf, 0, 2, w[0:10])

#- Get params from qin, run ex2d, put results into qout
def wrap_ex2d(x):
    i, specmin, nspec, wave = x
    return ex2d(image, imageivar, psf, specmin, nspec, wave)

#- Setup sub extractions
extract_args = list()
iarg = 0
for specmin in range(0, psf.nspec, opts.bundlesize):
    for i in range(0, len(w), opts.numwave):
        x = (iarg, specmin, opts.bundlesize, w[i:i+opts.numwave])
        extract_args.append(x)
        iarg += 1

t2 = time.time()
print('setup time {:.1f}'.format(t2-t1))

print('Running on {}/{} with {} logical cores'.format(
    platform.node(), platform.processor(), mp.cpu_count()))
print("bundlesize {} numwave {}".format(opts.bundlesize, opts.numwave))
print("OMP_NUM_THREADS={}".format(os.getenv('OMP_NUM_THREADS')))
print('nproc time rate')
for nproc in ntest:
    nmax = 2*int(nproc)
    p = mp.Pool(int(nproc))

    #- Start processes
    t0 = time.time()
    #for x in extract_args[0:nmax]:
    #     print(x)
    results = p.map(wrap_ex2d, [x for x in extract_args[0:nmax]])
    t = time.time() - t0
    rate = nmax * opts.bundlesize * opts.numwave / t
    print("{:3} {:5.1f} {:5.1f}".format(nproc, t, rate), flush=True)
    
