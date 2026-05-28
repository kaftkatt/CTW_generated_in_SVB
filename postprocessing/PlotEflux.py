#!/usr/bin/env python
# coding: utf-8

# # Animation

# In[1]:

import SVBfunc 

import xarray as xr
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import pylab as pl
from math import radians, cos
from scipy.integrate import trapezoid


coast='smooth'
tstart=2

depin=0

dirn = '/home/athelandersson/NETCDFs/' + str(coast) + '_NO/'
dirw = '/home/athelandersson/NETCDFs/' + str(coast) + '/'

dsw, dsn = SVBfunc.loadNetCDFs(dirw, dirn, 'dynVars',tstart)

dswPhi, dsnPhi = SVBfunc.loadNetCDFs(dirw, dirn, 'phiHyd',tstart)

pathVav='/home/athelandersson/CTW-analysis/Files/' + str(coast)+ '/MeanUVEL.nc'
dsVav=xr.open_dataset(pathVav)

pathPav='/home/athelandersson/CTW-analysis/Files/' + str(coast)+ '/MeanPHIHYD.nc'
dsPav=xr.open_dataset(pathPav)


avP=dsPav.avePHIHYD.values
avV=dsVav.aveUVEL.values

def fluxInternalWave(p,vel,dz,mask):
    Flux=(((p[:-1]*vel[:-1])+(p[1:]*vel[1:]))/2)*dz
    return Flux.sum(axis=0)

params = {'font.size': 22,
          'figure.figsize': (12, 8),
         'font.family':'sans'}
pl.rcParams.update(params)

#Index to call from the list of netcdfs
ind=0 #0 is day 2-3, 1 is day 3-4 until index 7 (day 9-10)
dep=0  #483.2 meter depth is the 55th element
t=0
tt=(((72*ind+t)*20)+2880)/60 # Gives amount of hours from start of the model, starts at hour 48 if ind=0 and t=0


depth=(dswPhi[0].Depth.values).astype(np.float32)
Z=(dswPhi[ind].Z.values).astype(np.float32)

LON=dswPhi[ind].XC-360
LAT=dswPhi[ind].YC

hFacCw = dsnPhi[ind].hFacC
hFacCusew=(hFacCw.values).astype(np.float32)

hfa = np.ma.masked_values(hFacCusew, 0)
maskw = np.ma.getmask(hfa)

Zmat=np.ma.masked_array(np.tile(Z,(len(maskw[0,0,:]),len(maskw[0,:,0]),1)).T,mask=maskw)
dz=Zmat[1:]-Zmat[:-1]

timeout=np.arange(2880,14400,20)
print(len(timeout))

fluxout=np.zeros((576,len(LAT),len(LON)))
for i in np.arange(0,576,1):
    t=i
    vel= SVBfunc.get_snapshot_at_level( t,depin,dsw,dsn,'UVELiw') 
    phi=SVBfunc.get_snapshot_at_level( t,depin,dswPhi,dsnPhi,'PHIHYDiw')
    phianoma=phi-avP
    velanoma=vel-avV
    print(t)
    fluxout[i,:,:]=fluxInternalWave((np.ma.masked_array(phianoma,mask=maskw)),(np.ma.masked_array(velanoma,mask=maskw)),dz,maskw)



pathE='/home/athelandersson/CTW-analysis/Files/' + str(coast)+ '/EfluxU.nc'
dsE = xr.Dataset({"Energyflux": (("time","lat","lon"), np.array(fluxout)),
					    },
				   coords ={
					 "lon" : LON.values,
					 "lat": LAT.values,
					 "time": np.array(timeout)
				      },
				       )

dsE.to_netcdf(pathE)




