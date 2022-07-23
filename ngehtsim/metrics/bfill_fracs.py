# -*- coding: utf-8 -*-
"""
Created on Sun Jul 08 12:39:01 2018
Updated on Wed Sep 22 2021

@authors: Daniel, Dom
"""
from __future__ import division
import ehtim as eh
from scipy.signal import convolve2d, fftconvolve
from scipy.stats import binned_statistic_2d
import numpy as np
import matplotlib
import os

def unit_disk(radius, N):
    """
    Creates a 2D array containing a disk of radius @radius with unit area.
    The dimension of the overall array is NxN.
    """
    
    out = np.zeros((N,N))
    ivec_1d = np.arange(0,N,1.0)
    jvec_1d = np.arange(0,N,1.0)
    ivec,jvec = np.meshgrid(ivec_1d,jvec_1d)
    row = ivec - int(N//2)
    col = jvec - int(N//2)
    index = (np.sqrt(row**2 + col**2) < radius)
    out[index] = 1.0 / index.sum()
    
    return out

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
    out[index] = 1.0
    
    return out

def grid_obs(obs, N, psize, stokes='I'):
    """
    Grids the u-v points in obs into an NxN square grid.
    Values are variance-weighted SNR, passed through a logistic function
    @obs: instance of obsdata object
    @N: integer, assumed to be odd
    """
    
    uvals = np.concatenate((obs.data['u'],-obs.data['u']))
    vvals = np.concatenate((obs.data['v'],-obs.data['v']))
    sigmas = np.concatenate((obs.data['sigma'],obs.data['sigma']))

    if (stokes == 'I'):
        datas = np.concatenate((obs.data['vis'],np.conj(obs.data['vis'])))
    elif (stokes == 'Q'):
        datas = np.concatenate((obs.data['qvis'],np.conj(obs.data['qvis'])))
    elif (stokes == 'U'):
        datas = np.concatenate((obs.data['uvis'],np.conj(obs.data['uvis'])))
    elif (stokes == 'V'):
        datas = np.concatenate((obs.data['vvis'],np.conj(obs.data['vvis'])))

    x = np.round(uvals/psize)
    y = np.round(vvals/psize)

    z = np.abs(datas) / (sigmas**2.0)
    znorm = 1.0 / (sigmas**2.0)

    outtemp, dum1, dum2, dum3 = binned_statistic_2d(x,y,z,statistic='sum',bins=int(1.2*N),range=[[-int(1.2*N)/2,int(1.2*N)/2],[-int(1.2*N)/2,int(1.2*N)/2]])
    outnorm, dum1, dum2, dum3 = binned_statistic_2d(x,y,znorm,statistic='sum',bins=int(1.2*N),range=[[-int(1.2*N)/2,int(1.2*N)/2],[-int(1.2*N)/2,int(1.2*N)/2]])
    outnorm[outnorm == 0.0] = 100.0*np.max(outnorm)
    outamp = outtemp / outnorm
    outsigma = np.sqrt(1.0/outnorm)
    out = outamp/outsigma
    out = np.transpose(out)

    return out

def logistic(values,logmid=1.5,logwid=0.525):
    """
    Transforms the input according to a logarithmic logistic function in
    
    values : input values to transform
    logmid : logarithmic midpoint of the logistic function
    logwid : logarithmic width of the logistic function
    """

    out = np.zeros_like(values)
    index = (values > 0.0)

    midpoint = 10.0**logmid
    term = ((values[index]/midpoint)**(1.0/logwid))
    out[index] = term / (1.0 + term)

    return out

def obs_fill(obs, longest, fov=100, N=10,logmid=1.5,logwid=0.525,stokes='I'):
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
    
    outconv = fftconvolve(grid_obs(obs, Npix, psize, stokes=stokes), unit_disk(conv_radius,int(round(2*conv_radius+1))),mode='full')
    outconv[outconv < 0.0] = 0.0
    outlog = logistic(outconv,logmid=logmid,logwid=logwid)

    total_area = np.sum(disk(fill_radius, len(outlog)))
    outret = np.sum(outlog) / total_area

    return outret
