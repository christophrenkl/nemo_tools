#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import modules
import xarray as xr
import numpy as np
from collections import namedtuple

"""
Module nemo_tools

Date: 2017-07-18

Description:
"""

__author__ = 'Christoph Renkl (christoph.renkl@dal.ca)'

# Define functions below. -----------------------------------------------------


def nemo_grid(fname):
    '''Read NEMO coordinates.nc file.'''    
    
    print('Read NEMO coordinates.nc file.')
    
    # load file into xarray
    NEMO = xr.open_dataset(fname)
    
    # coordinate limits
    NEMO['lonmin'] = NEMO.glamt.min()
    NEMO['lonmax'] = NEMO.glamu.max()
    NEMO['latmin'] = NEMO.gphit.min()
    NEMO['latmax'] = NEMO.gphiv.max()
    
    return NEMO
    
    
def nemo_angle(cfile):
    """
       SUBROUTINE angle
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE angle  ***
      !! 
      !! ** Purpose :   Compute angles between model grid lines and the North direction
      !!
      !! ** Method  :
      !!
      !! ** Action  :   Compute (gsint, gcost, gsinu, gcosu, gsinv, gcosv, gsinf, gcosf) arrays:
      !!      sinus and cosinus of the angle between the north-south axe and the 
      !!      j-direction at t, u, v and f-points
      !!
      !! History :
      !!   7.0  !  96-07  (O. Marti )  Original code
      !!   8.0  !  98-06  (G. Madec )
      !!   8.5  !  98-06  (G. Madec )  Free form, F90 + opt.
      !!   9.2  !  07-04  (S. Masson)  Add T, F points and bugfix in cos lateral boundary
      !!----------------------------------------------------------------------
    """
    
    # load coordinates.nc file
    ds = xr.open_dataset(cfile)
    
    # grid dimensions
    jpj, jpi = np.shape(ds.glamt)

    # ============================= !
    # Compute the cosinus and sinus !
    # ============================= !
    # (computation done on the north stereographic polar plane)

    # initialize output arrays
    gsint = np.zeros((jpj, jpi))
    gcost = np.zeros((jpj, jpi))    
    gsinu = np.zeros((jpj, jpi))
    gcosu = np.zeros((jpj, jpi))    
    gsinf = np.zeros((jpj, jpi))
    gcosf = np.zeros((jpj, jpi))    
    gsinv = np.zeros((jpj, jpi))
    gcosv = np.zeros((jpj, jpi))
    
    # degree to radians conversion
    rpi = np.pi
    rad = np.pi / 180.
    
    # north pole direction & modulous (at t-point)
    zlam = ds.glamt.values[1:-1, 1:]
    zphi = ds.gphit.values[1:-1, 1:]
    zxnpt = 0. - 2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    zynpt = 0. - 2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    znnpt = zxnpt*zxnpt + zynpt*zynpt

    # north pole direction & modulous (at u-point)
    zlam = ds.glamu.values[1:-1, 1:]
    zphi = ds.gphiu.values[1:-1, 1:]
    zxnpu = 0. - 2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    zynpu = 0. - 2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    znnpu = zxnpu*zxnpu + zynpu*zynpu

    # north pole direction & modulous (at v-point)
    zlam = ds.glamv.values[1:-1, 1:]
    zphi = ds.gphiv.values[1:-1, 1:]
    zxnpv = 0. - 2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    zynpv = 0. - 2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    znnpv = zxnpv*zxnpv + zynpv*zynpv

    # north pole direction & modulous (at f-point)
    zlam = ds.glamf.values[1:-1, 1:]
    zphi = ds.gphif.values[1:-1, 1:]
    zxnpf = 0. - 2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    zynpf = 0. - 2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )
    znnpf = zxnpf*zxnpf + zynpf*zynpf

    # j-direction: v-point segment direction (around t-point)
    zlam = ds.glamv.values[1:-1, 1:]
    zphi = ds.gphiv.values[1:-1, 1:]
    zlan = ds.glamv.values[:-2, 1:]
    zphh = ds.gphiv.values[:-2, 1:]
    zxvvt =  2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.cos( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    zyvvt =  2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.sin( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    znvvt = np.sqrt( znnpt * ( zxvvt*zxvvt + zyvvt*zyvvt ) )
    znvvt = np.maximum( znvvt, 1.e-14 )
    
    # j-direction: f-point segment direction (around u-point)
    zlam = ds.glamf.values[1:-1, 1:]
    zphi = ds.gphif.values[1:-1, 1:]
    zlan = ds.glamf.values[:-2, 1:]
    zphh = ds.gphif.values[:-2, 1:]
    zxffu =  2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.cos( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    zyffu =  2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.sin( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    znffu = np.sqrt( znnpu * ( zxffu*zxffu + zyffu*zyffu )  )
    znffu = np.maximum( znffu, 1.e-14 )

    # i-direction: f-point segment direction (around v-point)
    zlam = ds.glamf.values[1:-1, 1:]
    zphi = ds.gphif.values[1:-1, 1:]
    zlan = ds.glamf.values[1:-1, :-1]
    zphh = ds.gphif.values[1:-1, :-1]
    zxffv =  2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.cos( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    zyffv =  2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.sin( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    znffv = np.sqrt( znnpv * ( zxffv*zxffv + zyffv*zyffv )  )
    znffv = np.maximum( znffv, 1.e-14 )

    # j-direction: u-point segment direction (around f-point)
    zlam = ds.glamu.values[2: , 1:]
    zphi = ds.gphiu.values[2: , 1:]
    zlan = ds.glamu.values[1:-1, 1:]
    zphh = ds.gphiu.values[1:-1, 1:]
    zxuuf =  2. * np.cos( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.cos( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    zyuuf =  2. * np.sin( rad*zlam ) * np.tan( rpi/4. - rad*zphi/2. )   \
          -  2. * np.sin( rad*zlan ) * np.tan( rpi/4. - rad*zphh/2. )
    znuuf = np.sqrt( znnpf * ( zxuuf*zxuuf + zyuuf*zyuuf )  )
    znuuf = np.maximum( znuuf, 1.e-14 )

    # cosinus and sinus using scalar and vectorial products
    gsint[1:-1, 1:] = ( zxnpt*zyvvt - zynpt*zxvvt ) / znvvt
    gcost[1:-1, 1:] = ( zxnpt*zxvvt + zynpt*zyvvt ) / znvvt

    gsinu[1:-1, 1:] = ( zxnpu*zyffu - zynpu*zxffu ) / znffu
    gcosu[1:-1, 1:] = ( zxnpu*zxffu + zynpu*zyffu ) / znffu

    gsinf[1:-1, 1:] = ( zxnpf*zyuuf - zynpf*zxuuf ) / znuuf
    gcosf[1:-1, 1:] = ( zxnpf*zxuuf + zynpf*zyuuf ) / znuuf

    # (caution, rotation of 90 degres)
    gsinv[1:-1, 1:] = ( zxnpv*zxffv + zynpv*zyffv ) / znffv
    gcosv[1:-1, 1:] =-( zxnpv*zyffv - zynpv*zxffv ) / znffv

    #  =============== !
    # Geographic mesh !
    # =============== !

# CR: Currently not needed...
#      DO jj = 2, jpjm1
#         DO ji = fs_2, jpi   ! vector opt.
#            IF( MOD( ABS( glamv(ji,jj) - glamv(ji,jj-1) ), 360. ) < 1.e-8 ) THEN
#               gsint(ji,jj) = 0.
#               gcost(ji,jj) = 1.
#            ENDIF
#            IF( MOD( ABS( glamf(ji,jj) - glamf(ji,jj-1) ), 360. ) < 1.e-8 ) THEN
#               gsinu(ji,jj) = 0.
#               gcosu(ji,jj) = 1.
#            ENDIF
#            IF(      ABS( gphif(ji,jj) - gphif(ji-1,jj) )         < 1.e-8 ) THEN
#               gsinv(ji,jj) = 0.
#               gcosv(ji,jj) = 1.
#            ENDIF
#            IF( MOD( ABS( glamu(ji,jj) - glamu(ji,jj+1) ), 360. ) < 1.e-8 ) THEN
#               gsinf(ji,jj) = 0.
#               gcosf(ji,jj) = 1.
#            ENDIF
#         END DO
#      END DO

    # =========================== !
    # Lateral boundary conditions !
    # =========================== !

    # copy second to first row
    gsint[0,:] = gsint[1,:]
    gcost[0,:] = gcost[1,:]
    gsinu[0,:] = gsinu[1,:]
    gcosu[0,:] = gcosu[1,:]
    gsinf[0,:] = gsinf[1,:]
    gcosf[0,:] = gcosf[1,:]
    gsinv[0,:] = gsinv[1,:]
    gcosv[0,:] = gcosv[1,:]

    # copy second last to last row
    gsint[-1,:] = gsint[-2,:]
    gcost[-1,:] = gcost[-2,:]
    gsinu[-1,:] = gsinu[-2,:]
    gcosu[-1,:] = gcosu[-2,:]
    gsinf[-1,:] = gsinf[-2,:]
    gcosf[-1,:] = gcosf[-2,:]
    gsinv[-1,:] = gsinv[-2,:]
    gcosv[-1,:] = gcosv[-2,:]

    # copy second to first column
    gsint[:,0] = gsint[:,1]
    gcost[:,0] = gcost[:,1]
    gsinu[:,0] = gsinu[:,1]
    gcosu[:,0] = gcosu[:,1]
    gsinf[:,0] = gsinf[:,1]
    gcosf[:,0] = gcosf[:,1]
    gsinv[:,0] = gsinv[:,1]
    gcosv[:,0] = gcosv[:,1]

    # copy second last to last column
    gsint[:,-1] = gsint[:,-2]
    gcost[:,-1] = gcost[:,-2]
    gsinu[:,-1] = gsinu[:,-2]
    gcosu[:,-1] = gcosu[:,-2]
    gsinf[:,-1] = gsinf[:,-2]
    gcosf[:,-1] = gcosf[:,-2]
    gsinv[:,-1] = gsinv[:,-2]
    gcosv[:,-1] = gcosv[:,-2]
    
    # create named tuple to make handling of return values easier
    grid_angle = namedtuple('grid_angle', ["gsint", "gcost",
                                           "gsinu", "gcosu", 
                                           "gsinf", "gcosf",
                                           "gsinv", "gcosv"])

    # return values
    return grid_angle(gsint, gcost, gsinu, gcosu, gsinf, gcosf, gsinv, gcosv)

