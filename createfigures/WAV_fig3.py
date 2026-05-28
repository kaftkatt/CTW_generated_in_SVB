#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xarray as xr
import cmocean
import pylab as pl



# In[4]:


coast = 'smooth'

i=0
path='../datafiles/'+ str(coast)+'/WaveValues.nc'

ds  = xr.open_dataset(path)

W1=ds.WVELday2.values
W2=ds.WVELday4.values

maskW = ds.maskWVEL.values

eta1=ds.ETAday2.values
eta2=ds.ETAday4.values


maskETA = ds.maskETA.values

LON=ds.lon.values
LAT=ds.lat.values

depth=ds.depth.values

coast = 'realistic'

pathO='../datafiles/'+ str(coast)+'/WaveValues.nc'

dsO  = xr.open_dataset(pathO)

W1o=dsO.WVELday2.values
W2o=dsO.WVELday4.values


maskWo = dsO.maskWVEL.values

eta1o=dsO.ETAday2.values
eta2o=dsO.ETAday4.values

maskETAo = dsO.maskETA.values

LONo=dsO.lon.values
LATo=dsO.lat.values

deptho=dsO.depth.values



# In[5]:


params = {'font.size': 8,
          'figure.figsize': (7.48, 4.5),
         'font.family':'sans'}
pl.rcParams.update(params)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# In[7]:


fig = plt.figure()
gs = GridSpec(nrows=2, ncols=4,wspace=0.15,hspace=0.05)

vmin=-0.15
vmax=0.15

padi=0.2

xlab='Longitude [°]'
ylab='Latitude [°]'

ax = fig.add_subplot(gs[0,0]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LON,LAT,np.ma.masked_array(eta1*1000, mask=maskETA),cmap=cmocean.cm.curl,vmin=vmin,vmax=vmax)
ax.contour(LON,LAT,depth,  colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])


divider = make_axes_locatable(ax)
axdiv = divider.new_vertical(size = '5%', pad = padi)
fig.add_axes(axdiv)
cbar_ax = plt.colorbar(cax, cax=axdiv,orientation='horizontal')
cbar_ax.ax.xaxis.set_label_position("top")
cbar_ax.set_label('SSH [mm]')

ax.tick_params(axis='x',which='both', bottom=True, top=False, labelbottom=False)
ax.set( ylabel=ylab)

ax.text(-0.1,1.25, '(a)', transform=ax.transAxes)
ax.text(0.6,0.73, f'Smooth \nSurface \nDay 2', transform=ax.transAxes,horizontalalignment='left')

ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)

ax = fig.add_subplot(gs[1,0]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LON,LAT,np.ma.masked_array(eta2*1000, mask=maskETA),cmap=cmocean.cm.curl,vmin=vmin,vmax=vmax)
ax.contour(LON,LAT,depth,  colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])

ax.set(ylabel=ylab,xlabel=xlab)


ax.text(-0.1,1.05, '(e)', transform=ax.transAxes)
ax.text(0.6,0.73, f'Smooth \nSurface \nDay 4', transform=ax.transAxes,horizontalalignment='left')

ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)

ax = fig.add_subplot(gs[0,1]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LONo,LAT,np.ma.masked_array(eta1o*1000, mask=maskETAo),cmap=cmocean.cm.curl,vmin=vmin,vmax=vmax)
ax.contour(LONo,LATo,deptho,  colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])


divider = make_axes_locatable(ax)
axdiv = divider.new_vertical(size = '5%', pad = padi)
fig.add_axes(axdiv)
cbar_ax = plt.colorbar(cax, cax=axdiv,orientation='horizontal')
cbar_ax.ax.xaxis.set_label_position("top")
cbar_ax.set_label('SSH [mm]')

ax.tick_params(axis='x',which='both', bottom=True, top=False, labelbottom=False)
ax.tick_params(axis='y',which='both', left=False, labelleft=False)

ax.text(-0.1,1.25, '(b)', transform=ax.transAxes)
ax.text(0.6,0.73, f'Realistic \nSurface \nDay 2', transform=ax.transAxes,horizontalalignment='left')

ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)

ax = fig.add_subplot(gs[1,1]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LONo,LATo,np.ma.masked_array(eta2o*1000, mask=maskETAo),cmap=cmocean.cm.curl,vmin=vmin,vmax=vmax)
ax.contour(LONo,LATo,deptho,  colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])


ax.tick_params(axis='y',which='both', left=False, labelleft=False)
ax.set(xlabel=xlab)


ax.text(-0.1,1.05, '(f)', transform=ax.transAxes)
ax.text(0.6,0.73, f'Realistic \nSurface \nDay 4', transform=ax.transAxes,horizontalalignment='left')

ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)


vmin=-0.000002*1e6
vmax=0.000002*1e6


ax = fig.add_subplot(gs[0,2]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LON,LAT,np.ma.masked_array(W1, mask=maskW)*1e6,cmap=cmocean.cm.balance,vmin=vmin,vmax=vmax)
ax.contour(LON,LAT,depth, colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])


divider = make_axes_locatable(ax)
axdiv = divider.new_vertical(size = '5%', pad = padi)
fig.add_axes(axdiv)
cbar_ax = plt.colorbar(cax, cax=axdiv,orientation='horizontal')
cbar_ax.ax.xaxis.set_label_position("top")
cbar_ax.set_label('Vertical velocity\n [$10^{-6}$ ms$^{-1}$]')

ax.tick_params(axis='y',which='both', left=True, right=False, labelleft=False) 
ax.tick_params(axis='x',which='both', bottom=True, top=False, labelbottom=False)

ax.text(0.6,0.73, f'Smooth \n480 m \nDay 2', transform=ax.transAxes,horizontalalignment='left')
ax.text(-0.1,1.25, '(c)', transform=ax.transAxes)

ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)

ax = fig.add_subplot(gs[1,2]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LON,LAT,np.ma.masked_array(W2, mask=maskW)*1e6,cmap=cmocean.cm.balance,vmin=vmin,vmax=vmax)
ax.contour(LON,LAT,depth, colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])

ax.tick_params(axis='y',which='both', left=True, right=False, labelleft=False) 
ax.set(xlabel=xlab)

ax.text(0.6,0.73, f'Smooth \n480 m \nDay 4', transform=ax.transAxes,horizontalalignment='left')
ax.text(-0.1,1.05, '(g)', transform=ax.transAxes)


ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)

ax = fig.add_subplot(gs[0,3]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LONo,LATo,np.ma.masked_array(W1o, mask=maskWo)*1e6,cmap=cmocean.cm.balance,vmin=vmin,vmax=vmax)
ax.contour(LONo,LATo,deptho, colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])


divider = make_axes_locatable(ax)
axdiv = divider.new_vertical(size = '5%', pad = padi)
fig.add_axes(axdiv)
cbar_ax = plt.colorbar(cax, cax=axdiv,orientation='horizontal')
cbar_ax.ax.xaxis.set_label_position("top")
cbar_ax.set_label('Vertical velocity\n [$10^{-6}$ ms$^{-1}$]')

ax.tick_params(axis='y',which='both', left=True, right=False, labelleft=False) 
ax.tick_params(axis='x',which='both', bottom=True, top=False, labelbottom=False)

ax.text(0.6,0.73, f'Realistic \n480 m \nDay 2', transform=ax.transAxes,horizontalalignment='left')
ax.text(-0.1,1.25, '(d)', transform=ax.transAxes)

ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)

ax = fig.add_subplot(gs[1,3]) 

ax.set_facecolor('tan')
cax = ax.pcolormesh(LON,LAT,np.ma.masked_array(W2o, mask=maskWo)*1e6,cmap=cmocean.cm.balance,vmin=vmin,vmax=vmax)
ax.contour(LONo,LATo,deptho, colors=['0.2','0.6'], 
                levels=[0,500],linewidths=[1,0.8])

ax.tick_params(axis='y',which='both', left=True, right=False, labelleft=False) 

ax.set(xlabel=xlab)
ax.text(0.6,0.73, f'Realistic \n480 m \nDay 4', transform=ax.transAxes,horizontalalignment='left')
ax.text(-0.1,1.05, '(h)', transform=ax.transAxes)


ax.set_xlim(-122,-114) 
ax.set_ylim(27,35.3)
ax.set_aspect(1)

plt.tight_layout()
plt.savefig('../figures/WAV_fig3.pdf')


# In[ ]:




