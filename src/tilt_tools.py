#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import modules
import glob
import cv2
import os
import netCDF4 as nc
import numpy as np
import plot_maps as pm
import matplotlib.pyplot as plt
import scipy.interpolate as scint
from math import radians, cos, sin, asin, sqrt
from mpl_toolkits.basemap import Basemap
from cmocean import cm

"""
Module tilt_tools.py

Date:

Description:
"""

__author__ = 'Christoph Renkl (christoph.renkl@dal.ca)'

# Define functions below. -----------------------------------------------------


def coastal_wet_points(lmask):
    "Extract alongshore wet points from land mask."
    
    
    # invert landmask and and convert to uint8
    lmask_inv = ~lmask.astype(np.uint8)

    # get contours of coastline
    ret, thresh = cv2.threshold(lmask_inv, 254, 255, cv2.THRESH_BINARY)
    img, cnts, hierarchy = cv2.findContours(thresh,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    
    # pick the largest contour
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    cnts = np.squeeze(cnts)
    
    nj, ni = np.shape(lmask)
    
    cnts = cnts[(cnts[:,0] != 1   ) &  
                (cnts[:,0] != ni-2) & 
                (cnts[:,1] != 1   ) &
                (cnts[:,1] != nj-2)]
    
#CR 2017-08-23-----------------------------------------------------------------
#    print(conts)
#    
#    # find indices along domain boundary --------------------------------------
#    
#    nj, ni = np.shape(lmask)    
#    splitinds = np.empty(0).astype(np.int32)
#    
#    # left boundary (first columns = 1)
#    tmp = np.where(conts[:, 0] == 1)[0]
#    if np.size(tmp) > 0:
#        splitinds = np.concatenate((splitinds, np.unique(tmp[[0, -1]])))
#    
#    # right boundary - I probably only need this when open
#    tmp = np.where(conts[:, 0] == ni-2)[0]
#    if np.size(tmp) > 0:
#        splitinds = np.concatenate((splitinds, np.unique(tmp[[0, -1]])))
#    
#    # lower boundary
#    tmp = np.where(conts[:, 1] == 1)[0]    
#    print(tmp)
#    if np.size(tmp) > 0:
#        splitinds = np.concatenate((splitinds, np.unique(tmp[[0, -1]])))
#    
#    # upper boundary
#    tmp = np.where(conts[:, 1] == nj-2)[0]
#    if np.size(tmp) > 0:
#        splitinds = np.concatenate((splitinds, np.unique(tmp[[0, -1]])))
#        
#    splitinds = np.sort(splitinds)
#    splitinds = splitinds[np.where((splitinds != 0) &
#                                   (splitinds != len(conts)-1))]
#    
#    # split contours
#    conts = np.split(conts, splitinds)
# -----------------------------------------------------------------------------

   
#   This is old code and can be deletede eventually, but check first if it
#   works for a full model domain.
# 
#    # find indices along domain boundary
#    splitinds = np.where(conts[:,0] == 1)[0][[[0,-1]]]
#    splitinds = np.append(splitinds,
#                          np.where(conts[:, 0] == conts[:,0].max())[0][[0,-1]])
#    splitinds = np.append(splitinds, np.where(conts[:,1] == 1)[0][[0,-1]])
#    splitinds = np.append(splitinds,
#                          np.where(conts[:,1] == conts[:,1].max())[0][[0,-1]])
#    splitinds = np.sort(splitinds)
#    splitinds = splitinds[np.where((splitinds != 0) &
#                                   (splitinds != len(conts)-1))]
#    # split contours
#    conts = np.split(conts, splitinds)

    # return
    return cnts


def alongshore_distance(ipind, e1u, e2v):
    "Compute alongshore distance from horizontal scale factors."

    # number of alongshore wet points
    nn = np.shape(ipind)[0]

    # initialization
    iold, jold = ipind[0,:]
    ds = np.array([])

    # compute distance between two points
    for inc in range(1, nn):

        inew, jnew = ipind[inc,:]

        if inew == iold and jnew == (jold + 1):

            dist = e2v[jold, iold]

        elif inew == iold and jnew == (jold - 1):

            dist = e2v[jnew, inew]

        elif inew == (iold + 1) and jnew == jold:

            dist = e1u[jold, iold]

        elif inew == (iold - 1) and jnew == jold:

            dist = e1u[jnew, inew]

        elif inew == (iold + 1) and jnew == (jold + 1):

            dist = np.sqrt(e1u[jold,iold]**2 + e2v[jold,inew]**2)

        elif inew == (iold - 1) and jnew == (jold - 1):

            dist = np.sqrt(e1u[jold,inew]**2 + e2v[jnew,inew]**2)

        elif inew == (iold + 1) and jnew == (jold - 1):

            dist = np.sqrt(e1u[jnew,iold]**2 + e2v[jnew,iold]**2)

        elif inew == (iold - 1) and jnew == (jold + 1):

            dist = np.sqrt(e1u[jnew,inew]**2 + e2v[jold,iold]**2)

        else:

            print('Something went wrong!')

        # swap indices
        iold = inew
        jold = jnew

        # fill array of increment distances
        ds = np.append(ds, dist)

    # compute alongshore distance
    asdst = np.cumsum(np.append(0, ds))

    return ds, asdst


def alongshore_distance_lonlat(aslon, aslat):
    "Compute alongshore distance from horizontal scale factors."
    
    # number of alongshore wet points
    nn = np.shape(aslon)[0]

    # initialization
    ds = np.array([])

    # compute distance between two points
    for ic in np.arange(1, nn):
        
        # compute distance between coordinates
        dist = haversine(aslon[ic - 1], aslat[ic - 1], aslon[ic], aslat[ic])

        # fill array of increment distances
        ds = np.append(ds, dist)

    # compute alongshore distance
    asdst = np.cumsum(np.append(0, ds))
    
    return asdst


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # Earth's radius [m]
    R = 6371e3

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    d = 2 * R * np.arcsin(np.sqrt(hav))

    return d
    

def alongshore_steric_height(mdir, runID, asind):
    """
    Compute mean alongshore steric height.
    """
    
    # parameters and constants ------------------------------------------------

    g = 9.80665  # acceleration due to gravity [m s-2]
    
    # load data ---------------------------------------------------------------
 
    mesh = nc.Dataset(mdir+'/Configuration/mesh_mask.nc')

    # get mask at T-points
    tmask = np.squeeze(mesh.variables['tmask'][:])
    
    # close file
    mesh.close()
    
    # get data file name
    flist = sorted(glob.glob(mdir+'/Output/'+runID+'/*_grid_T_*.nc'))
    nfiles = len(flist)
    
    # number of alongshore grid points
    npts = np.shape(asind)[0]

    # initialize array
    assterhgt = np.empty((0, npts))
    
    # loop over all files
    for ifile in np.arange(nfiles):

        # file name
        fname = flist[ifile]
        
        # load file
        data = nc.Dataset(fname)
        
        # get time information
        time = data.variables['time_counter'][:]
        
        # initialize arrays
        astmp = np.empty([np.shape(time)[0], 0])
        
        for ipt in np.arange(npts):
            
            iind, jind = asind[ipt,:]
            
            rhd = np.ma.masked_array(data.variables['rhd'][..., jind, iind]).T
            e3t = np.ma.masked_array(data.variables['e3t'][..., jind, iind]).T
            
            # mask arrays
            rhd[tmask[..., jind, iind] == 0] = np.ma.masked
            e3t[tmask[..., jind, iind] == 0] = np.ma.masked
            
            # compute steric height
            tmp = - np.ma.sum( np.ma.multiply(rhd, e3t), axis=0)
            astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
    
        data.close()
            
        # alongshore component
        assterhgt = np.ma.append(assterhgt, astmp, axis=0)
        
        # compute mean
        assterhgt_mean = np.mean(assterhgt, axis=0)
        
    return assterhgt_mean
    

def alongshore_mssh(mdir, runID, asind):
    """
    Load alongshore mean dynamic topography (MDT).
    """ 
    # get data file name
    flist = sorted(glob.glob(mdir+'/Output/'+runID+'/*_grid_2D_*.nc'))
    nfiles = len(flist)
    
    # number of alongshore grid points
    npts, dummy = np.shape(asind)

    # initialize array
    asssh = np.empty((0, npts))
    
    # loop over all files
    for ifile in np.arange(nfiles):

        # file name
        fname = flist[ifile]
        
        # load file
        data = nc.Dataset(fname)
        
        # get time information
        time = data.variables['time_counter'][:]
        
        # initialize arrays
        astmp = np.empty([np.shape(time)[0], 0])
#        astmp = np.array([])
        
        for ipt in np.arange(npts):
            
            iind, jind = asind[ipt,:]
            
            tmp = np.ma.masked_array(data.variables['ssh'][..., jind, iind]).T
            
            astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
    
        data.close()

        # alongshore component
        asssh = np.ma.append(asssh, astmp, axis=0)
        
        # compute mean
        asmssh = np.mean(asssh, axis=0)
        
    return asmssh
    

def mean_ssh(mdir, runID):
    
    flist = sorted(glob.glob(mdir+'/Output/'+runID+'/*_grid_2D_*.nc'))
    nfiles = len(flist)

    # initialize array
    ssh = []
    
    # loop over all files
    for ifile in np.arange(nfiles):

        # file name
        fname = flist[ifile]
        
        # load file
        data = nc.Dataset(fname)
        
        tmp = data.variables['ssh'][:]
            
        ssh.append(tmp)
    
        data.close()
        
    # stack all data files
    ssh = np.vstack(ssh)
    
    # compute mean sea surface height
    mssh = np.mean(ssh, axis=0)
    
    return mssh
    

def alongshore_component(mdir, runID, asind, comp):
    """
    Extract alongshore component of momentum equation.
    """
    
    # parameters and constants ------------------------------------------------

    g = 9.80665  # acceleration due to gravity [m s-2]
        
    print(comp)
    
    # load masks --------------------------------------------------------------
    
    # load mesh mask file    
    mesh = nc.Dataset(mdir+'/Configuration/mesh_mask.nc')

    # get horizontal scale factors
    e1u = np.squeeze(mesh.variables['e1u'][:])
    e2v = np.squeeze(mesh.variables['e2v'][:])

    # get mask at U- and V-points
    umask = np.squeeze(mesh.variables['umask'][:])
    vmask = np.squeeze(mesh.variables['vmask'][:])

    # get bottom layer indices
    mbathy = np.squeeze(mesh.variables['mbathy'][:])
    
    # close mesh mask file
    mesh.close()
    
    # load data ---------------------------------------------------------------    
    
    # list all data files name and count them
    flist = sorted(glob.glob(mdir+'/Output/'+runID+'/*_%s_*.nc' % comp.upper() ))
    nfiles = 1 # len(flist)
    
    nt = 2
    
    # number of alongshore grid points
    npts = np.shape(asind)[0]
    
    # make sure component is lower case
    comp = comp.lower()
    
    # initialize array
    ascomp = np.empty((0, npts))   
    
    # loop over all files
    for ifile in  np.arange(nfiles):

        # get file name and load file
        fname = flist[ifile]
        data = nc.Dataset(fname)
        
        # get time information
        time = data.variables['time_counter'][:nt]
        
        # get last part of file name to ensure equivalent files are loaded
        fname.split('-')
            
        # get U-grid file name
        uflist = glob.glob(mdir+'/Output/'+runID+'/*_grid_U_*'+fname.split('-')[1])
        ufname = uflist[0]
    
        # load U-gridfile
        udata = nc.Dataset(ufname)
        
        # get V-grid file name
        vflist = glob.glob(mdir+'/Output/'+runID+'/*_grid_V_*'+fname.split('-')[1])
        vfname = vflist[0]  
    
        # load V-grid file
        vdata = nc.Dataset(vfname)
        
        # initialize arrays
        astmp = np.zeros([np.shape(time)[0], 1])
        
        # first indices of alongshore wet points
        iold, jold = asind[0,:]
        
        # loop over all alongshore wet points
        for ipt in np.arange(1, npts):
            
            # next indices of alongshore wet points
            inew, jnew = asind[ipt,:]
            
            print('old: ', iold, jold)
            print('new: ', inew, jnew)               
    
            if inew == iold and jnew == (jold + 1):
                print('move north')
                
                # read V-grid data
                vcomp = np.ma.masked_array(
                        data.variables['vtrd_'+comp][:nt, ..., jold, iold]).T            
                e3v = np.ma.masked_array(
                      vdata.variables['e3v'][:nt, ..., jold, iold]).T
                
                # mask vertical scale factors
                e3v[vmask[..., jold, iold] == 0] = np.ma.masked
                
                # compute local water depth
                vdepth = np.ma.sum(e3v, axis=0)
                      
                if comp == 'tau':
                    
                    e3v[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    vcomp[vmask[..., jold, iold] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3v)
                    
                    # bottom indices, remember Python is zero-based
                    ikbv = np.minimum( mbathy[jold+1, iold],
                                       mbathy[jold, iold] ) - 1
                    
                    # mask all, but bottom cell
                    e3v[np.arange(nk)!=ikbv] = np.ma.masked
                
                # depth-averaging
                vtmp = np.ma.sum( np.ma.multiply(vcomp, e3v), axis=0)            
                vdavg = np.ma.divide(vtmp, vdepth)
                
                # alongshore component
                tmp = vdavg
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )

#                print(e3v[ikbv])
#                print('vdepth', vdepth)
    
            elif inew == iold and jnew == (jold - 1):
                print('move south')
                
                # read V-grid data
                vcomp = np.ma.masked_array(
                        data.variables['vtrd_'+comp][:nt, ..., jnew, inew]).T            
                e3v = np.ma.masked_array(
                      vdata.variables['e3v'][:nt, ..., jnew, inew]).T
                
                # mask vertical scale factors
                e3v[vmask[..., jnew, inew] == 0] = np.ma.masked
                
                # compute local water depth
                vdepth = np.ma.sum(e3v, axis=0)
                      
                if comp == 'tau':
                    
                    e3v[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    vcomp[vmask[..., jnew, inew] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3v)
                    
                    # bottom indices, remember Python is zero-based
                    ikbv = np.minimum( mbathy[jnew+1, inew],
                                       mbathy[jnew, inew] ) - 1
                    
                    # mask all, but bottom cell
                    e3v[np.arange(nk)!=ikbv] = np.ma.masked
                
                # depth-averaging
                vtmp = np.ma.sum( np.ma.multiply(vcomp, e3v), axis=0)            
                vdavg = np.ma.divide(vtmp, vdepth)
                
                # alongshore component
                tmp = -vdavg
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
                              
#                print(e3v[ikbv])
#                print('vdepth', vdepth)
    
            elif inew == (iold + 1) and jnew == jold:
                print('move east')
                
                # read U-grid data
                ucomp = np.ma.masked_array(
                        data.variables['utrd_'+comp][:nt, ..., jold, iold]).T            
                e3u = np.ma.masked_array(
                      udata.variables['e3u'][:nt, ..., jold, iold]).T
                
                # mask vertical scale factors
                e3u[umask[..., jold, iold] == 0] = np.ma.masked
                
                # compute local water depth
                udepth = np.ma.sum(e3u, axis=0)
                      
                if comp == 'tau':
                    
                    e3u[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    ucomp[umask[..., jold, iold] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3u)
                    
                    # bottom indices, remember Python is zero-based
                    ikbu = np.minimum( mbathy[jold, iold+1],
                                       mbathy[jold, iold] ) - 1
                    
                    # mask all, but bottom cell
                    e3u[np.arange(nk)!=ikbu] = np.ma.masked
                
                # depth-averaging
                utmp = np.ma.sum( np.ma.multiply(ucomp, e3u), axis=0)            
                udavg = np.ma.divide(utmp, udepth)
                
                # alongshore component
                tmp = udavg
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
                              
#                print(e3u[ikbu])
#                print('udepth', udepth)
    
            elif inew == (iold - 1) and jnew == jold:
                print('move west')
                
                # read U-grid data
                ucomp = np.ma.masked_array(
                        data.variables['utrd_'+comp][:nt, ..., jnew, inew]).T            
                e3u = np.ma.masked_array(
                      udata.variables['e3u'][:nt, ..., jnew, inew]).T
                      
                ucur = np.ma.masked_array(
                      udata.variables['uo'][..., jnew, inew]).T
                
                # mask vertical scale factors
                e3u[umask[..., jnew, inew] == 0] = np.ma.masked
                
                # compute local water depth
                udepth = np.ma.sum(e3u, axis=0)
                      
                if comp == 'tau':
                    
                    e3u[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    ucomp[umask[..., jnew, inew] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3u)
                    
                    # bottom indices, remember Python is zero-based
                    ikbu = np.minimum( mbathy[jnew, inew+1],
                                       mbathy[jnew, inew] ) - 1
                    
                    # mask all, but bottom cell
                    print(np.shape(e3u))
                    e3u[np.arange(nk)!=ikbu,:] = np.ma.masked
                    
#                    fig = plt.figure(figsize=(14, 8)) 
#                    h = plt.imshow(ucur, interpolation='none',
#                                   cmap=cm.balance,
#                                   vmin=-0.05, vmax=0.05)
#                    fig.colorbar(h)
                    
#                print('utrd_bfr: ', np.mean(ucomp[ikbu]))
#                print('u_bot: ', np.mean(ucur[ikbu]))
#                print('e3u: ', np.mean(e3u[ikbu]))
                
                # depth-averaging
                utmp = np.ma.sum( np.ma.multiply(ucomp, e3u), axis=0)            
                udavg = np.ma.divide(utmp, udepth)
                
                # alongshore component
                tmp = -udavg

#                print('ikbu:', ikbu)
                
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
                              
#                print(e3u[ikbu])
#                print('udepth', udepth)
                
            elif inew == (iold + 1) and jnew == (jold + 1):
                print('move northeast')
                
                # read U-grid data
                ucomp = np.ma.masked_array(
                        data.variables['utrd_'+comp][:nt, ..., jold, iold]).T            
                e3u = np.ma.masked_array(
                      udata.variables['e3u'][:nt, ..., jold, iold]).T
                
                # mask vertical scale factors
                e3u[umask[..., jold, iold] == 0] = np.ma.masked
                
                # compute local water depth
                udepth = np.ma.sum(e3u, axis=0)
                      
                if comp == 'tau':
                    
                    e3u[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    ucomp[umask[..., jold, iold] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3u)
                    
                    # bottom indices, remember Python is zero-based
                    ikbu = np.minimum( mbathy[jold, iold+1],
                                       mbathy[jold, iold] ) - 1
                    
                    # mask all, but bottom cell
                    e3u[np.arange(nk)!=ikbu] = np.ma.masked
                
                # depth-averaging
                utmp = np.ma.sum( np.ma.multiply(ucomp, e3u), axis=0)            
                udavg = np.ma.divide(utmp, udepth)
                
                # read V-grid data
                vcomp = np.ma.masked_array(
                        data.variables['vtrd_'+comp][:nt, ..., jold, inew]).T            
                e3v = np.ma.masked_array(
                      vdata.variables['e3v'][:nt, ..., jold, inew]).T
                
                # mask vertical scale factors
                e3v[vmask[..., jold, inew] == 0] = np.ma.masked
                
                # compute local water depth
                vdepth = np.ma.sum(e3v, axis=0)
                      
                if comp == 'tau':
                    
                    e3v[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    vcomp[vmask[..., jold, inew] == 0] = np.ma.masked
                    
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3v)
                    
                    # bottom indices, remember Python is zero-based
                    ikbv = np.minimum( mbathy[jold+1, inew],
                                       mbathy[jold, inew] ) - 1
                    
                    # mask all, but bottom cell
                    e3v[np.arange(nk)!=ikbv,] = np.ma.masked
                
                # depth-averaging
                vtmp = np.ma.sum( np.ma.multiply(vcomp, e3v), axis=0)            
                vdavg = np.ma.divide(vtmp, vdepth)

                dist = np.sqrt(e1u[jold,iold]**2 + e2v[jold,inew]**2)
                
                # alongshore component
                tmp = ( udavg * e1u[jold, iold]
                      + vdavg * e2v[jold, inew] ) / dist
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
                              
#                print(e3u[ikbu])
#                print('udepth', udepth)
#                print(e3v[ikbv])
#                print('vdepth', vdepth)
                
            elif inew == (iold - 1) and jnew == (jold - 1):
                print('move southwest')
                
                # read U-grid data
                ucomp = np.ma.masked_array(
                        data.variables['utrd_'+comp][:nt, ..., jold, inew]).T            
                e3u = np.ma.masked_array(
                      udata.variables['e3u'][:nt, ..., jold, inew]).T
                
                # mask vertical scale factors
                e3u[umask[..., jold, inew] == 0] = np.ma.masked
                
                # compute local water depth
                udepth = np.ma.sum(e3u, axis=0)
                      
                if comp == 'tau':
                    
                    e3u[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    ucomp[umask[..., jold, inew] == 0] = np.ma.masked
                    
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3u)
                    
                    # bottom indices, remember Python is zero-based
                    ikbu = np.minimum( mbathy[jold, inew+1],
                                       mbathy[jold, inew] ) - 1
                    
                    # mask all, but bottom cell
                    e3u[np.arange(nk)!=ikbu] = np.ma.masked
                
                # depth-averaging
                utmp = np.ma.sum( np.ma.multiply(ucomp, e3u), axis=0)            
                udavg = np.ma.divide(utmp, udepth)
                
                # read V-grid data
                vcomp = np.ma.masked_array(
                        data.variables['vtrd_'+comp][:nt, ..., jnew, inew]).T            
                e3v = np.ma.masked_array(
                      vdata.variables['e3v'][:nt, ..., jnew, inew]).T
                
                # mask vertical scale factors
                e3v[vmask[..., jnew, inew] == 0] = np.ma.masked
                
                # compute local water depth
                vdepth = np.ma.sum(e3v, axis=0)
                      
                if comp == 'tau':
                    
                    e3v[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    vcomp[vmask[..., jnew, inew] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3v)
                    
                    # bottom indices, remember Python is zero-based
                    ikbv = np.minimum( mbathy[jnew+1, inew],
                                       mbathy[jnew, inew] ) - 1
                    
                    # mask all, but bottom cell
                    e3v[np.arange(nk)!=ikbv] = np.ma.masked
                
                # depth-averaging
                vtmp = np.ma.sum( np.ma.multiply(vcomp, e3v), axis=0)            
                vdavg = np.ma.divide(vtmp, vdepth)

                dist = np.sqrt(e1u[jold,inew]**2 + e2v[jnew,inew]**2)
                
                # alongshore component
                tmp = ( - udavg * e1u[jold, inew]
                        - vdavg * e2v[jnew, inew] ) / dist
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
                              
#                print(e3u[ikbu])
#                print('udepth', udepth)
#                print(e3v[ikbv])
#                print('vdepth', vdepth)
    
            elif inew == (iold + 1) and jnew == (jold - 1):
                print('move southeast')
                
                # read U-grid data
                ucomp = np.ma.masked_array(
                        data.variables['utrd_'+comp][:nt, ..., jnew, iold]).T            
                e3u = np.ma.masked_array(
                      udata.variables['e3u'][:nt, ..., jnew, iold]).T
                
                # mask vertical scale factors
                e3u[umask[..., jnew, iold] == 0] = np.ma.masked
                
                # compute local water depth
                udepth = np.ma.sum(e3u, axis=0)
                      
                if comp == 'tau':
                    
                    e3u[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    ucomp[umask[..., jnew, iold] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3u)
                    
                    # bottom indices, remember Python is zero-based
                    ikbu = np.minimum( mbathy[jnew, iold+1],
                                       mbathy[jnew, iold] ) - 1
                    
                    # mask all, but bottom cell
                    e3u[np.arange(nk)!=ikbu] = np.ma.masked
                
                # depth-averaging
                utmp = np.ma.sum( np.ma.multiply(ucomp, e3u), axis=0)            
                udavg = np.ma.divide(utmp, udepth)
                
                # read V-grid data
                vcomp = np.ma.masked_array(
                        data.variables['vtrd_'+comp][:nt, ..., jnew, iold]).T            
                e3v = np.ma.masked_array(
                      vdata.variables['e3v'][:nt, ..., jnew, iold]).T
                
                # mask vertical scale factors
                e3v[vmask[..., jnew, iold] == 0] = np.ma.masked
                
                # compute local water depth
                vdepth = np.ma.sum(e3v, axis=0)
                      
                if comp == 'tau':
                    
                    e3v[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    vcomp[vmask[..., jnew, iold] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3v)
                    
                    # bottom indices, remember Python is zero-based
                    ikbv = np.minimum( mbathy[jnew+1, iold],
                                       mbathy[jnew, iold] ) - 1
                    
                    # mask all, but bottom cell
                    e3v[np.arange(nk)!=ikbv] = np.ma.masked
                
                # depth-averaging
                vtmp = np.ma.sum( np.ma.multiply(vcomp, e3v), axis=0)            
                vdavg = np.ma.divide(vtmp, vdepth)

                dist = np.sqrt(e1u[jnew,iold]**2 + e2v[jnew,iold]**2)
                
                # alongshore component
                tmp = ( udavg * e1u[jnew, iold]
                      - vdavg * e2v[jnew, iold] ) / dist
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
                              
#                print(e3u[ikbu])
#                print('udepth', udepth)
#                print(e3v[ikbv])
#                print('vdepth', vdepth)
    
            elif inew == (iold - 1) and jnew == (jold + 1):
                print('move northwest')
                
                # read U-grid data
                ucomp = np.ma.masked_array(
                        data.variables['utrd_'+comp][:nt, ..., jnew, inew]).T            
                e3u = np.ma.masked_array(
                      udata.variables['e3u'][:nt, ..., jnew, inew]).T
                
                # mask vertical scale factors
                e3u[umask[..., jnew, inew] == 0] = np.ma.masked
                
                # compute local water depth
                udepth = np.ma.sum(e3u, axis=0)
                      
                if comp == 'tau':
                    
                    e3u[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    ucomp[umask[..., jnew, inew] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3u)
                    
                    # bottom indices, remember Python is zero-based
                    ikbu = np.minimum( mbathy[jnew, inew+1],
                                       mbathy[jnew, inew] ) - 1
                    
                    # mask all, but bottom cell
                    e3u[np.arange(nk)!=ikbu] = np.ma.masked
                
                # depth-averaging
                utmp = np.ma.sum( np.ma.multiply(ucomp, e3u), axis=0)            
                udavg = np.ma.divide(utmp, udepth)
                
                # read V-grid data
                vcomp = np.ma.masked_array(
                        data.variables['vtrd_'+comp][:nt, ..., jold, iold]).T            
                e3v = np.ma.masked_array(
                      vdata.variables['e3v'][:nt, ..., jold, iold]).T
                
                # mask vertical scale factors
                e3v[vmask[..., jold, iold] == 0] = np.ma.masked
                
                # compute local water depth
                vdepth = np.ma.sum(e3v, axis=0)
                      
                if comp == 'tau':
                    
                    e3v[1:] = np.ma.masked
                    
                else:
                    
                    # mask arrays
                    vcomp[vmask[..., jold, iold] == 0] = np.ma.masked
                
                if comp == 'bfr':
                    
                    # number of vertical levels
                    nk, nt = np.shape(e3v)
                    
                    # bottom indices, remember Python is zero-based
                    ikbv = np.minimum( mbathy[jold+1, iold],
                                       mbathy[jold, iold] ) - 1
                    
                    # mask all, but bottom cell
                    e3v[np.arange(nk)!=ikbv] = np.ma.masked
                
                # depth-averaging
                vtmp = np.ma.sum( np.ma.multiply(vcomp, e3v), axis=0)            
                vdavg = np.ma.divide(vtmp, vdepth)

                dist = np.sqrt(e1u[jnew,inew]**2 + e2v[jold,iold]**2)
                
                # alongshore component
                tmp = ( - udavg * e1u[jnew, inew]
                        + vdavg * e2v[jold, iold] ) / dist
                astmp = np.ma.append( astmp, tmp[:,np.newaxis], axis=1 )
                              
#                print(e3u)
#                print('udepth', udepth)
#                print(e3v)
#                print('vdepth', vdepth)
    
            else:
    
                print('Something went wrong!')
                
            # swap indices   
            iold, jold = inew, jnew
            
        # close all data files
        udata.close()
        vdata.close()
        data.close()
        
        ascomp = np.ma.append(ascomp, astmp, axis=0)
    
    # compute mean
    ascomp_mean = np.mean(astmp, axis=0)
    
    return ascomp_mean

        
        
## set to True if you want to re-extract tilts
#clean = True
#
#tiltfile = 'mdt_tilts.npz'
#
#figdir = '/home/chrenkl/Projects/MDT_tilt/Figures/for_Phil'
#
#nt_test = 1000 #8760
#
#if not os.path.exists(figdir):    
#    os.makedirs(figdir)
#
#
#if clean is True or not os.path.isfile(tiltfile):
#    
#    print('clean')
#
#    mdir = '/data/po/jasket/chrenkl/Models/GoMSS'
#
#    grid = nc.Dataset(mdir+'/Configuration/bathy_meter.nc')
#
#    lons = grid.variables['nav_lon'][:]
#    lats = grid.variables['nav_lat'][:]
#    bathy = grid.variables['Bathymetry'][:]
#
#    # close file
#    grid.close()
#
#    # create land mask
#    lmask = np.ma.getmask(bathy)
#
#    # find alongshore wet points and select coastline of interest
#    conts = coastal_wet_points(lmask)
#    ipind = conts[1]
#
#    # get coordinates of alongshore wet points
#    iplon = lons[ipind[:, 1], ipind[:, 0]]
#    iplat = lats[ipind[:, 1], ipind[:, 0]]
#
#    mesh = nc.Dataset(mdir+'/Configuration/mesh_mask.nc')
#
#    # get mesh mask
#    tmask = mesh.variables['tmask'][0, 0, :, :]  # surface tmask
#
#    # get horizontal scale factors
#    e1u = np.squeeze(mesh.variables['e1u'][:])
#    e2v = np.squeeze(mesh.variables['e2v'][:])
#
#    # close file
#    mesh.close()
#
#    # compute alongshore distance
#    ds, asdst = alongshore_distance(ipind, e1u, e2v)
#
#    # total length of alongshore integration path [m]
#    iplen = asdst[-1]
#
#    # GoMSS MDT
#    mrun = 'namelist'
#
#    flist = glob.glob(mdir+'/Output/'+mrun+'/*_grid_2D.nc')
#    fname = flist[0]
#
#    data = nc.Dataset(fname)
#
#    tsteps = data.variables['time_counter'][:]
#    tsteps = nc.num2date(tsteps, units=data.variables['time_counter'].units)
#
#    ssh_daily = data.variables['ssh'][:nt_test, ...]
#
#    data.close()
#
#    mssh_gomss = np.mean(ssh_daily, axis=0)
#    mssh_gomss = np.ma.masked_where(tmask == 0, mssh_gomss)
#
#    mdt_gomss = mssh_gomss - np.ma.mean(mssh_gomss)
#
##    asmdt_gomss = mssh_gomss[ipind[:,1], ipind[:,0]]
##
##    # sample interval and points along integration path [m]
##    sint = np.round((iplen/25)*1e-3)*1e3
##    spts = np.arange(0, iplen, sint)\
##
##    # get HYCOM/NCODA  data
##    hdir = '/data/po/jasket/chrenkl/Reanalysis/HYCOM_NCODA'
##    hrun = 'expt_19.1'
##    
##    flist = glob.glob(hdir+'/'+hrun+'/*_surf_el_*.nc')
##    fname = flist[0]
##
##    hdata = nc.Dataset(fname)
##
##    # get coordinates (disregard some latitudes to find coastal wet points)
##    hlon = hdata.variables['lon'][:]
##    hlat = hdata.variables['lat'][:180]
##
##    # get sea surface height
##    ssh = hdata.variables['surf_el'][:,:180,:]
##
##    hdata.close()
##    
##    # compute mean sea surface height
##    mssh = np.mean(ssh, axis=0)
##
##    # get land mask
##    hmask = np.ma.getmask(mssh)
##
##    # find coastal wet points
##    hconts = coastal_wet_points(hmask)
##    hcinds = hconts[1]
##    hclon = hlon[hcinds[:,0]]
##    hclat = hlat[hcinds[:,1]]
##
##    # initialize some arrays
##    slon = np.array([])
##    slat = np.array([])
##    sdst = np.array([])
##
##    hslon = np.array([])
##    hslat = np.array([])
##    hasmdt = np.array([])
##
##    # get coordinates and distance of integration path closest to sample points
##    for ipt, point in enumerate(spts):
##        idx = (np.abs(asdst - point)).argmin()
##        slon = np.append(slon, iplon[idx])
##        slat = np.append(slat, iplat[idx])
##        sdst = np.append(sdst, asdst[idx])
##
##        # index of minimum great circle distance to HYCOM coastal wet point
##        hidx = (haversine(slon[ipt], slat[ipt], hclon, hclat)).argmin()
##
##        # HYCOM data closest to sample point
##        hslat = np.append(hslat, hclat[hidx])
##        hslon = np.append(hslon, hclon[hidx])
##        hasmdt = np.append(hasmdt, mssh[hcinds[hidx][1], hcinds[hidx][0]])
#
#    # tide gauge data
#    obslon = np.asarray([-71.0533, -70.2467, -66.9817, -66.1333, -63.5833, -60.2500])
#    obslat = np.asarray([ 42.3533,  43.6567,  44.9033,  43.8333,  44.6667,  46.2167])
#    obsmdt = np.asarray([-.31, -.34, -.34, -.32, -.3, -.29])
#
##    obsdst = np.array([])
##    gomss_at_obs = np.array([])
##    hycom_at_obs = np.array([])
##
##    for iobs in range(len(obsmdt)):
##        obsidx = (haversine(obslon[iobs], obslat[iobs], iplon, iplat)).argmin()
##        obsdst = np.append(obsdst, asdst[obsidx])
##
##        # GoMSS at tide gauge location
##        gomss_at_obs = np.append(gomss_at_obs, asmdt_gomss[obsidx])
##
##        # HYCOM at tide gauge location
##        hycom_obsidx = (haversine(iplon[obsidx], iplat[obsidx], hclon, hclat)).argmin()
##        hycom_at_obs = np.append(hycom_at_obs,
##                        mssh[hcinds[hycom_obsidx][1], hcinds[hycom_obsidx][0]])
##
##    # remove mean
##    asmdt_obs = obsmdt - np.mean(obsmdt)
##    asmdt_gomss = np.asarray(asmdt_gomss - np.mean(gomss_at_obs))
##    asmdt_hycom = hasmdt - np.mean(hycom_at_obs)
#    
#    # surface currents:
#    flist = glob.glob(mdir+'/Output/'+mrun+'/*1h*_grid_U.nc')
#    fname = flist[0]
#
#    udata = nc.Dataset(fname)
#    
#    flist = glob.glob(mdir+'/Output/'+mrun+'/*1h*_grid_V.nc')
#    fname = flist[0]
#
#    vdata = nc.Dataset(fname)
#    
#    sossu = udata.variables['sossu']
#    sossv = vdata.variables['sossv']
#    
#    nt, nj, ni = np.shape(sossu)
#    
#    nt = nt_test
#    
##    usqr = np.zeros((nj-1, ni-1))
#    usqmean = np.zeros((nj, ni))
#    vsqmean = np.zeros((nj, ni))
#    
#    for it in np.arange(nt):
#        
#        usqmean = np.add(usqmean, np.square(sossu[it,...]))        
#        vsqmean = np.add(vsqmean, np.square(sossv[it,...]))
#        
#        
##        # unstaggering
##        utmp = np.add(np.ma.masked_values(sossu[it,..., :-1], 0),
##                      np.ma.masked_values(sossu[it,..., 1:], 0) ) / 2
##        
##        vtmp = np.add(np.ma.masked_values(sossv[it,..., :-1, :], 0),
##                      np.ma.masked_values(sossv[it,..., 1:, :], 0) ) / 2       
##        
##        usqr = np.add( usqr, np.square(utmp[:-1,:]), np.square(vtmp[:,:-1])) 
#    
#    
#    vdata.close()
#    udata.close()
#    
#    
#    
##    umsqr = np.square(umean/nt)    
##    vmsqr = np.square(umean/nt)
#    
#    utmp = np.add(usqmean[..., :-1]/nt, usqmean[..., 1:]/nt) / 2
#    vtmp = np.add(vsqmean[..., :-1, :]/nt, vsqmean[..., 1:, :]/nt) / 2
#    
#    usqr = np.add(utmp[:-1, ...], vtmp[..., :-1])
#    
##    usqr = usqr/nt
#    
##    np.savez(tiltfile,
##             lons,
##             lats,
##             np.asarray(mdt_gomss),
##             tmask,
##             asmdt_gomss,
##             asmdt_hycom,
##             asmdt_obs,
##             obsdst)
#
#else:
#
#    print('load')
#
#    np.load(tiltfile)
#
## zoom into Bay of Fundy
#zrows, zcols = np.where((lons > -68.) & (lons < -62.) & (lats > 43) & (lats < 46))
#
#mdt_gomss_zbof = mssh_gomss[zrows.min():zrows.max(), zcols.min():zcols.max()]
#
#lons_bof = lons[zrows.min():zrows.max(), zcols.min():zcols.max()]
#lats_bof = lats[zrows.min():zrows.max(), zcols.min():zcols.max()]
#
#g = 9.80665  # acceleration due to gravity [m s-2]
#
## Bernoulli set-down
#set_down = usqr / (2*g)
#set_down_zbof = set_down[zrows.min():zrows.max(), zcols.min():zcols.max()]
#
#mdt_diff = mdt_gomss_zbof + set_down_zbof
#
## remove means
#mdt_gomss_zbof = mdt_gomss_zbof - np.mean(mdt_gomss_zbof)
#mdt_diff = mdt_diff - np.mean(mdt_diff)
#
#pmin = - np.max(np.maximum(np.absolute(mdt_gomss_zbof),
#                           np.absolute(set_down_zbof),
#                           np.absolute(mdt_diff)))
#
#pmax = np.max(np.maximum(np.absolute(mdt_gomss_zbof),
#                         np.absolute(set_down_zbof),
#                         np.absolute(mdt_diff)))
#
#
### PLOTTING
##
### plots for Phil
##
#### figure 1: GoMSS and observations, full y-axis
#### ------------------------------------------------------------------------------
###fig = plt.figure(figsize=(12, 2.85))
###ax1 = fig.add_subplot(111)
###
###plt.plot(asdst*1e-3, asmdt_gomss, '-b',
###         linewidth=2,
###         color=[0, 0.4470, 0.7410],
###         label='GoMSS (w/ tides)')
###plt.plot(obsdst*1e-3, asmdt_obs, ':ok',label='Observations')
###
###plt.axis([0, (iplen*1e-3).max(), -.2, .1])
###plt.legend(loc=4)
###
####labels
###ax1.set_xlabel("Alongshore Distance [km]")
###ax1.set_ylabel("MDT [m]")
###
#### second x-axis
###ax2 = ax1.twiny()
###ax2.set_xticks(obsdst*1e-3)
###ax2.set_xticklabels(['Boston', 'Portland', 'Eastport', 'Yarmouth', 'Halifax', 
###           'North Sydney'])
###
#### save figure
###pm.save_pdf(figdir+'/asmdt_gomss_obs')
###
###
#### figure 2: GoMSS and observations, clipped y-axis
#### ------------------------------------------------------------------------------
###fig = plt.figure(figsize=(12, 2.85))
###ax1 = fig.add_subplot(111)
###
###plt.plot(asdst*1e-3, asmdt_gomss, '-b',
###         linewidth=2,
###         color=[0, 0.4470, 0.7410],
###         label='GoMSS (w/ tides)')
###plt.plot(obsdst*1e-3, asmdt_obs, ':ok',label='Observations')
###
###plt.axis([0, (iplen*1e-3).max(), -.05, .05])
###plt.legend(loc=4)
###
####labels
###ax1.set_xlabel("Alongshore Distance [km]")
###ax1.set_ylabel("MDT [m]")
###
#### second x-axis
###ax2 = ax1.twiny()
###ax2.set_xticks(obsdst*1e-3)
###ax2.set_xticklabels(['Boston', 'Portland', 'Eastport', 'Yarmouth', 'Halifax', 
###           'North Sydney'])
###
#### save figure
###pm.save_pdf(figdir+'/asmdt_gomss_obs_clipped')
###
#### figure 3: GoMSS MDT map, full domain
#### ------------------------------------------------------------------------------
###fig, map = pm.plot_varmap(lons, lats, mdt_gomss,'MDT', proj='merc')
###
###scat = map.scatter(obslon, obslat,
###                   latlon=True,
###                   s=20,
###                   marker='s', color='k',
###                   zorder=10)                    
###fig.set_size_inches(15, 8)
###
#### save figure
###pm.save_pdf(figdir+'/mdt_map_gomss')
##
### figure 4: GoMSS MDT map, zoom of Bay of Fundy
### ------------------------------------------------------------------------------
### create new figure
##fig = plt.figure()
##
### setup map
##map = Basemap(projection='merc',
##              llcrnrlat=43., urcrnrlat=46.,
##              llcrnrlon=-68., urcrnrlon=-62,
##              resolution='h')
##
### convert longitudes and latitudes to x and y
##xx, yy = map(lons_bof, lats_bof)
##
##clevs = np.arange(-.14, .14, .02)
##
### plot variable field on map
##mesh = map.pcolormesh(xx, yy, mdt_gomss_zbof,
##                    cmap=cm.balance)
##                      
##map.contour(xx, yy, mdt_gomss_zbof,
##            levels=[-.1, -0.05],
##            colors='k',
##            linestyles='--',
##            alpha=0.75,
##            linewidths=0.5)    
##            
##map.contour(xx, yy, mdt_gomss_zbof,
##            levels=[0, 0.05],
##            colors='k',
##            linestyles='-',
##            alpha=0.75,
##            linewidths=0.35)
##             
##scat = map.scatter(obslon, obslat,
##                   latlon=True,
##                   s=80,
##                   marker='s', color='k',
##                   zorder=10)                     
##  
##mesh.set_clim(vmin= - np.max(np.absolute(mdt_gomss_zbof)),
##              vmax=   np.max(np.absolute(mdt_gomss_zbof)))
##
### color bar
##cb = map.colorbar(mesh, size="4%", pad="2%")
##cb.set_label('Mean Dynamic Topography [m]')
##cb.ax.tick_params(labelsize=12)
##
### map appearance
##map.drawcoastlines(color=[.4, .4, .4], linewidth=1)
##map.fillcontinents(color=[.6, .6, .6])
##map.drawmapboundary(linewidth=2)
##map.drawparallels(np.arange(int(43.), int(47.)+1, 1),
##                  labels=[1, 0, 0, 0], linewidth=0.,
##                  fontsize=12)
##map.drawmeridians(np.arange(int(-68.), int(-62.)+1, 1),
##                  labels=[0, 0, 0, 1], linewidth=0.,
##                  fontsize=12)
##                  
##fig.set_size_inches(15, 8)
##
### save figure
##pm.save_pdf(figdir+'/mdt_map_gomss_zoom_bof')
#
## figure 5: GoMSS MDT & surface currents map, zoom of Bay of Fundy
## ------------------------------------------------------------------------------
## create new figure
#fig = plt.figure()
#
## setup map
#map = Basemap(projection='merc',
#              llcrnrlat=43., urcrnrlat=46.,
#              llcrnrlon=-68., urcrnrlon=-62,
#              resolution='h')
#
## convert longitudes and latitudes to x and y
#xx, yy = map(lons_bof, lats_bof)
#
#clevs = np.arange(-.14, .14, .02)
#
## plot variable field on map
#mesh = map.pcolormesh(xx, yy, mdt_gomss_zbof,
#                      cmap=cm.balance)
#                      
#map.contour(xx, yy, mdt_gomss_zbof,
#            levels=np.arange(-.2, 0, .02),
#            colors='k',
#            linestyles='--',
#            alpha=0.75,
#            linewidths=0.5)    
#            
#map.contour(xx, yy, mdt_gomss_zbof,
#            levels=np.arange(0, .2, .02),
#            colors='k',
#            linestyles='-',
#            alpha=0.75,
#            linewidths=0.35)
#             
#scat = map.scatter(obslon, obslat,
#                   latlon=True,
#                   s=80,
#                   marker='s', color='k',
#                   zorder=10)
#                   
#mesh.set_clim(vmin=pmin, vmax=pmax)
#
## color bar
#cb = map.colorbar(mesh, size="4%", pad="2%")
#cb.set_label('Mean Dynamic Topography [m]')
#cb.ax.tick_params(labelsize=12)
#
## map appearance
#map.drawcoastlines(color=[.4, .4, .4], linewidth=1)
#map.fillcontinents(color=[.6, .6, .6])
#map.drawmapboundary(linewidth=2)
#map.drawparallels(np.arange(int(43.), int(47.)+1, 1),
#                  labels=[1, 0, 0, 0], linewidth=0.,
#                  fontsize=12)
#map.drawmeridians(np.arange(int(-68.), int(-62.)+1, 1),
#                  labels=[0, 0, 0, 1], linewidth=0.,
#                  fontsize=12)
#                  
#fig.set_size_inches(15, 8)
#
## save figure
#pm.save_pdf(figdir+'/mdt_map_gomss_zoom_bof')
#
## figure 6: GoMSS Bernoulli set-down, zoom of Bay of Fundy
## ------------------------------------------------------------------------------
## create new figure
#fig = plt.figure()
#
## setup map
#map = Basemap(projection='merc',
#              llcrnrlat=43., urcrnrlat=46.,
#              llcrnrlon=-68., urcrnrlon=-62,
#              resolution='h')
#
## convert longitudes and latitudes to x and y
#xx, yy = map(lons_bof, lats_bof)
#
## plot variable field on map
#mesh = map.pcolormesh(xx, yy, set_down_zbof,
#                    cmap=cm.balance)
#                   
#mesh.set_clim(vmin=pmin, vmax=pmax)
#
## color bar
#cb = map.colorbar(mesh, size="4%", pad="2%")
#cb.set_label('Bernoulli Set-down [m]')
#cb.ax.tick_params(labelsize=12)
#
## map appearance
#map.drawcoastlines(color=[.4, .4, .4], linewidth=1)
#map.fillcontinents(color=[.6, .6, .6])
#map.drawmapboundary(linewidth=2)
#map.drawparallels(np.arange(int(43.), int(47.)+1, 1),
#                  labels=[1, 0, 0, 0], linewidth=0.,
#                  fontsize=12)
#map.drawmeridians(np.arange(int(-68.), int(-62.)+1, 1),
#                  labels=[0, 0, 0, 1], linewidth=0.,
#                  fontsize=12)
#                  
#fig.set_size_inches(15, 8)
#
## save figure
#pm.save_pdf(figdir+'/bernoulli_set-down_bof')
#
## figure 6: GoMSS MDT - Bernoulli set-down, zoom of Bay of Fundy
## ------------------------------------------------------------------------------
## create new figure
#fig = plt.figure()
#
## setup map
#map = Basemap(projection='merc',
#              llcrnrlat=43., urcrnrlat=46.,
#              llcrnrlon=-68., urcrnrlon=-62,
#              resolution='h')
#
## convert longitudes and latitudes to x and y
#xx, yy = map(lons_bof, lats_bof)
#
## plot variable field on map
#mesh = map.pcolormesh(xx, yy, mdt_diff,
#                    cmap=cm.balance)
#                    
#mesh.set_clim(vmin=pmin, vmax=pmax)
#                      
#map.contour(xx, yy, mdt_diff,
#            levels=np.arange(-.2, 0, .02),
#            colors='k',
#            linestyles='--',
#            alpha=0.75,
#            linewidths=0.5)    
#            
#map.contour(xx, yy, mdt_diff,
#            levels=np.arange(0, .2, .02),
#            colors='k',
#            linestyles='-',
#            alpha=0.75,
#            linewidths=0.35)
#
## color bar
#cb = map.colorbar(mesh, size="4%", pad="2%")
#cb.set_label('Bernoulli Set-down [m]')
#cb.ax.tick_params(labelsize=12)
#
## map appearance
#map.drawcoastlines(color=[.4, .4, .4], linewidth=1)
#map.fillcontinents(color=[.6, .6, .6])
#map.drawmapboundary(linewidth=2)
#map.drawparallels(np.arange(int(43.), int(47.)+1, 1),
#                  labels=[1, 0, 0, 0], linewidth=0.,
#                  fontsize=12)
#map.drawmeridians(np.arange(int(-68.), int(-62.)+1, 1),
#                  labels=[0, 0, 0, 1], linewidth=0.,
#                  fontsize=12)
#                  
#fig.set_size_inches(15, 8)
#
## save figure
#pm.save_pdf(figdir+'/mdt_minus_bernoulli_set-down_bof')
#
#
#### -----------------------------------------------------------------------------
###
###fig = plt.figure(figsize=(12, 2.85))
###plt.plot(asdst*1e-3, asmdt_gomss, '-b',label='GoMSS (w/ tides)')
###plt.plot(sdst*1e-3, asmdt_hycom, 'or',label='HYCOM/NCODA')
###plt.plot(obsdst*1e-3, asmdt_obs, ':ok',label='Observations')
###plt.axis([0, (iplen*1e-3).max(), -.2, .1])
###
###plt.legend(loc=3)
###
####labels
###plt.xlabel("Alongshore Distance [km]")
###plt.ylabel("MDT [m]")
###
###pm.save_pdf('../Figures/MDT_comparison')
###
###
###
###fig = plt.figure(figsize=(12, 2.85))
###plt.plot(asdst*1e-3, asmdt_gomss, '-b',label='GoMSS (w/ tides)')
###plt.plot(sdst*1e-3, asmdt_hycom, 'or',label='HYCOM/NCODA')
###plt.plot(obsdst*1e-3, asmdt_obs, ':ok',label='Observations')
###plt.axis([0, (iplen*1e-3).max(), -.05, .05])
###
###plt.legend(loc=4)
###
####labels
###plt.xlabel("Alongshore Distance [km]")
###plt.ylabel("MDT [m]")
###
###pm.save_pdf('../Figures/MDT_comparison_clipped')
###
##### quick plot of mean sea surface height
####fig = pm.plot_varmap(hlon, hlat, mssh,
####                          'Mean Sea Surface Height', proj='merc')
####
##### quick plot of mean sea surface height
####fig = pm.plot_varmap(lons, lats, mdt_gomss,
####                          'Mean Sea Surface Height', proj='merc')
####       