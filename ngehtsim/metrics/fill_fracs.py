"""
Created on Sun Jul 08 12:39:01 2018
@authors: Daniel, Dom
"""
from __future__ import division
import ehtim as eh
from scipy.signal import convolve2d, fftconvolve
import numpy as np
import matplotlib
from math import pi, sqrt
from ehtim.const_def import *
import os

def disk(radius, N):
    """
    Creates a 2D array containing a disk of radius @radius of value 1.
    The dimension of the overall array is NxN.
    """
    
    out = np.zeros((N,N))
    ivec_1d = np.arange(0,N,1.0)
    jvec_1d = np.arange(0,N,1.0)
    ivec,jvec = np.meshgrid(ivec_1d,jvec_1d)
    row = ivec - int(N//2)
    col = jvec - int(N//2)
    index = (np.sqrt(row**2 + col**2) < radius)
    out[index] = 1
    
    return out


def grid_obs(obs, N, psize):
    """
    Grids the u-v points in obs into an NxN square grid. Values are 1, otherwise zero.
    @obs: instance of obsdata object
    @N: integer, assumed to be odd
    """
    
    uvals = np.concatenate((obs.data['u'],-obs.data['u']))
    vvals = np.concatenate((obs.data['v'],-obs.data['v']))
    
    out = np.zeros((int(1.2*N),int(1.2*N)))#chosen to be larger to allow for large blurring
    x = np.round(uvals/psize)
    y = np.round(vvals/psize)
    out_temp, dum1, dum2 = np.histogram2d(x,y,bins=int(1.2*N),range=[[-int(1.2*N)/2,int(1.2*N)/2],[-int(1.2*N)/2,int(1.2*N)/2]])
    out[(out_temp != 0.0)] = 1
    out = np.transpose(out)
    
    return(out)

def radial_fill(grid, radius):
    """
    Returns the filling fraction inside a circle within a grid which is assumed to be square.
    @grid: the 2d grid, assumed to be square
    @radius: the radius in pixels of the internal circle
    """
    
    circ = disk(radius, len(grid))
    out = (np.sum((grid>.5)*circ)) / np.sum(circ)
    
    return out

def obs_fill(obs, longest, fov=100, N=10):
    """
    Calculates the filling fraction for an observation out to a given baseline @longest
    using field of view @fov. Automatically computes number of pixels so that there are
    @N pixels across a single convolution element.
    """
    
    conv_radius_pre = 0.71/(fov*eh.RADPERUAS)
    fill_radius_pre = longest
    
    psize = conv_radius_pre / (float(N)/2.0)
    
    conv_radius = conv_radius_pre / psize
    fill_radius = fill_radius_pre / psize
    
    Npix = np.ceil(2.0*fill_radius)
    
    out = fftconvolve(grid_obs(obs, Npix, psize), disk(conv_radius,int(round(2*conv_radius+1))),mode='full')    
    outret = radial_fill(out,fill_radius)
    
    return outret
