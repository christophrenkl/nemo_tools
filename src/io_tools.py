#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import modules
import os
import xarray as xr
import numpy as np

"""
Module  io_tools.py

Date: 2017-08-29

Description:
"""

__author__   = 'Christoph Renkl'
__email__    = 'christoph.renkl@dal.ca'

# Define functions below. -----------------------------------------------------

def main():
    ''' '''
    
    # runID
    runID = 'R001'
    
    # data directory
    nemodir = '/data/po/jasket/chrenkl/Models/MBnest'
    
    # list model output files
    files = os.path.join(nemodir, ('Output/%s' % runID), '*_grid_2D_*.nc')
    
    NEMO = mdt_from_2d_hourly(files)
    
    return NEMO
    
    
def mdt_from_2d_hourly(files, sshvar='ssh'):
    
    # load data file
    data = xr.open_mfdataset(files, chunks={'time_counter': 75},
                             concat_dim='time_counter')
    
    # create output dictionary
    NEMO = {'tlon'   : data[sshvar].nav_lon_grid_T.values,
            'tlat'   : data[sshvar].nav_lat_grid_T.values,
            'tlonbds': data['bounds_lon_grid_T'].values[0, ...],
            'tlatbds': data['bounds_lat_grid_T'].values[0, ...],
            'ni'     : data.dims['x_grid_T'],
            'nj'     : data.dims['y_grid_T']}
    
    # mean dynamic topography
    mssh = data[sshvar].mean(dim='time_counter').values
    NEMO['mdt'] = mssh - np.mean(mssh)
    
    return NEMO
    
    
def mean_surface_currents_from_2d_hourly(files, uvar='sossu', vvar='sossv'):
    
    # load data file
    data = xr.open_mfdataset(files, chunks={'time_counter': 75},
                             concat_dim='time_counter')
    
    # create output dictionary
    NEMO = {'tlon'   : data['nav_lon_grid_T'].values,
            'tlat'   : data['nav_lat_grid_T'].values,
            'tlonbds': data['bounds_lon_grid_T'].values[0, ...],
            'tlatbds': data['bounds_lat_grid_T'].values[0, ...],
            'ni'     : data.dims['x_grid_T'],
            'nj'     : data.dims['y_grid_T']}
    
    # surface currents
    us = data[uvar].values
    vs = data[vvar].values
    
    # unstagger current components to T-points
    ust = np.zeros_like(us)
    ust[..., :, 1:] = ( us[..., :  , :-1] + us[...,  :, 1:] ) / 2
    vst = np.zeros_like(vs)
    vst[..., 1:, :] = ( vs[..., :-1, :  ] + vs[..., 1:,  :] ) / 2
    
    # compute mean and add to output dictionary
    NEMO['must'] = ust.mean(axis=0)
    NEMO['mvst'] = vst.mean(axis=0)
    
    return NEMO
    
    
def tidal_elevation(fname, constituents=['M2', 'S2', 'N2', 'K1', 'O1', 'M4']):
    
    # load data file
    data = xr.open_dataset(fname)
    
    # create output dictionary
    NEMO = {'tlon'   : data['nav_lon'].values,
            'tlat'   : data['nav_lat'].values,
            'tlonbds': data['bounds_lon'].values,
            'tlatbds': data['bounds_lat'].values,
            'ni'     : data.dims['x'],
            'nj'     : data.dims['y']}
    
    # loop through constituents
    for const in constituents:
        
        # compute amplitude
        NEMO[const+'_amp'] = np.abs(data[const+'_eta_real'].values
                                    - 1j * data[const+'_eta_imag'].values)
        
        # compute phase and convert to degrees
        NEMO[const+'_pha'] = np.angle(data[const+'_eta_real'].values
                                      - 1j * data[const+'_eta_imag'].values)
        NEMO[const+'_pha'] = np.rad2deg(NEMO[const+'_pha'])
    
    return NEMO


if __name__ == '__main__':
    
    # run main method
    NEMO = main()
