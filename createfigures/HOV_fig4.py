#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import pylab as pl
from matplotlib.gridspec import GridSpec


# In[2]:


def plot_HOVMOLLER(ax,LON,TIME,VAL,title,ctitle,vmin,vmax,fig,lat,lon,lab,cbarall,nr):

    xlab='Time [days]'
    if lab==1:
        ylab='Distance [km]' #'Depth [m]'
    else:
        ylab=''

    ax.set(xlabel=xlab, ylabel=ylab)
    ax.set_xticks([2880, 4320, 5760, 7200, 8640, 10080, 11520, 12960, 14400])
    ax.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_title(title)

    if ctitle=='SSH [mm]':
        cax = ax.pcolormesh(TIME,LON,np.transpose(VAL),cmap=cmocean.cm.curl,vmin=vmin,vmax=vmax) 
    else:    
        cax = ax.pcolormesh(TIME,LON,np.transpose(VAL),cmap=cmocean.cm.balance,vmin=vmin,vmax=vmax) 

    if cbarall==1:
    ##FOR THE SAME COLORBAR FOR ALL OF THE PLOTS
        cbar_ax = fig.add_axes([0.95, 0.17, 0.03, 0.7])
        fig.colorbar(cax, cax=cbar_ax)
        cbar_ax.set_ylabel(ctitle)
    else:
    ##FOR A COLORBAR FOR EACH PLOT
        divider = make_axes_locatable(ax)
        axdiv = divider.new_vertical(size = '5%', pad = 0.5)
        fig.add_axes(axdiv)
        cbar_ax = plt.colorbar(cax, cax=axdiv,orientation='horizontal')
        cbar_ax.ax.xaxis.set_label_position("top")
        cbar_ax.set_label(ctitle)
    #ax.set_aspect(1./ax.get_data_ratio())
    ax.text(-0.15, 1.05, nr, fontweight='bold', color='k', 
        transform=ax.transAxes)



# # VELOCITY

# In[4]:


coast='../datafiles/smooth'
tstart=72

pathVEL=str(coast) + '/WVELAC.nc'
dsVEL= xr.open_dataset(pathVEL)

WVEL=dsVEL.Valfilt.values[tstart:]
distVEL=dsVEL.dist.values

TIMEVEL=dsVEL.time.values[tstart:]/60

lat_acVEL=dsVEL.latAC.values
lon_acVEL=dsVEL.lonAC.values

pathVELb=str(coast) + '/WVELAC_SVB.nc'
dsVELb= xr.open_dataset(pathVELb)

WVELb=dsVELb.Valfilt.values[tstart:]
distVELb=dsVELb.dist.values

TIMEVELb=dsVELb.time.values[tstart:]/60

lat_acVELb=dsVELb.latAC.values
lon_acVELb=dsVELb.lonAC.values


# In[5]:


coast='../datafiles/realistic'

pathVELOrg=str(coast) + '/WVELAC.nc'
dsVELOrg= xr.open_dataset(pathVELOrg)


WVELOrg=dsVELOrg.Valfilt.values
distVELOrg=dsVELOrg.dist.values

TIMEVELOrg=dsVELOrg.time.values

lat_acVELOrg=dsVELOrg.latAC.values
lon_acVELOrg=dsVELOrg.lonAC.values

pathVELbOrg=str(coast) + '/WVELAC_SVB.nc'
dsVELbOrg= xr.open_dataset(pathVELbOrg)

WVELbOrg=dsVELbOrg.Valfilt.values
distVELbOrg=dsVELbOrg.dist.values

TIMEVELbOrg=dsVELbOrg.time.values

lat_acVELbOrg=dsVELbOrg.latAC.values
lon_acVELbOrg=dsVELbOrg.lonAC.values


# In[6]:


params = {'font.size': 10,
          'figure.figsize': (7.48, 8),
         'font.family':'sans'}
pl.rcParams.update(params)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# In[7]:


end=np.where(lat_acVEL>33.5)[0][0]
endOrg=np.where(lat_acVELOrg>33.5)[0][0]


# In[8]:


fig = plt.figure()
gs = GridSpec(nrows=2, ncols=4, hspace=0.4,wspace=1)

ax = fig.add_subplot(gs[0, 0:2])
vmin=-5
vmax=5
cbarall=1

cax=plot_HOVMOLLER(ax,distVEL[:end],TIMEVELb,WVELb[:,:end]*1e6,'','Vertical velocity  [$10^{-6}$ ms$^{-1}$]',vmin,vmax,fig,lat_acVELb,lon_acVELb,1,cbarall,'(a)')

ax = fig.add_subplot(gs[0, 2:])

cax1=plot_HOVMOLLER(ax,distVELOrg[:endOrg],TIMEVELbOrg,WVELbOrg[:,:endOrg]*1e6,'','Vertical velocity  [$10^{-6}$ ms$^{-1}$]',vmin,vmax,fig,lat_acVELbOrg,lon_acVELbOrg,1,cbarall,'(b)')

ax.text(-0.45, 1.13, 'Run with SVB', color='k', fontsize=14,
transform=ax.transAxes)



ax = fig.add_subplot(gs[1, 0:2])
vmin=-5
vmax=5
cbarall=1

cax=plot_HOVMOLLER(ax,distVEL[:end],TIMEVEL,WVEL[:,:end]*1e6,'Smooth','Vertical velocity  [$10^{-6}$ ms$^{-1}$]',vmin,vmax,fig,lat_acVEL,lon_acVEL,1,cbarall,'(c)')

ax = fig.add_subplot(gs[1, 2:])

cax1=plot_HOVMOLLER(ax,distVELOrg[:endOrg],TIMEVELOrg,WVELOrg[:,:endOrg]*1e6,'Realistic','Vertical velocity  [$10^{-6}$ ms$^{-1}$]',vmin,vmax,fig,lat_acVELOrg,lon_acVELOrg,1,cbarall,'(d)')

ax.text(-0.8, 1.13, 'Run with SVB - Run without SVB', color='k',fontsize=14, 
transform=ax.transAxes)

plt.savefig('../figures/HOV_fig4.pdf',bbox_inches='tight')


