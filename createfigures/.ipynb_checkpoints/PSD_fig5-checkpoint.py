#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean
import xarray as xr
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat


# In[3]:
coasto='../datafiles/realistic'

pathVELo=str(coasto) + '/WVELAC.nc'
dsVELo= xr.open_dataset(pathVELo)
WVELo=dsVELo.Valfilt.values
distVELo=dsVELo.dist.values

TIMEVELo=dsVELo.time.values

lat_acVELo=dsVELo.latAC.values
lon_acVELo=dsVELo.lonAC.values


coast='../datafiles/smooth'
pathVEL=str(coast) + '/WVELAC.nc'
dsVEL= xr.open_dataset(pathVEL)

WVEL=dsVEL.Valfilt.values
distVEL=dsVEL.dist.values

TIMEVEL=dsVEL.time.values/60

lat_acVEL=dsVEL.latAC.values
lon_acVEL=dsVEL.lonAC.values


# In[4]:


def FFRQ(Wfilt,timemin,dist):

    times = (timemin)*60 #in s

    t0 = 172800 # start is day 2 

    dt = 1200 # 20 min 
    fs=1/1200

    nx = len(dist)
    nt = int(720/2+1)

    psd = np.zeros((nx,nt))
    phase = np.zeros((nx,nt))
    freq =  np.zeros((nx,nt))

    nr=720

    if len(Wfilt)>576: 
        start=72
    else: 
        start=144

    for ii in np.arange(nx): #nx
        arrin=np.zeros(nr)
        arrindif=np.zeros(nr)
        N = len(Wfilt[:,ii])

        arrin[start:]=Wfilt[:,ii]

        xdft = np.fft.rfft(arrin,n=nr) 
        xdft = xdft[0:int(nr/2+1)]
        psdx = (1/(fs*N)) * np.abs(xdft)**2 
        psdx[1:-1] = 2*psdx[1:-1]
        FFTfreq = np.fft.rfftfreq(nr, d=1200) #np.arange(0,fs/2+fs/N,fs/N)

        psd[ii,:] = psdx
        freq[ii,:] =  FFTfreq




    return psd, freq


# In[5]:


psdfilt, freqfilt=FFRQ(WVEL,TIMEVEL,distVEL) 
psdfilto, freqfilto=FFRQ(WVELo,TIMEVELo,distVELo) 


# In[6]:


def closest(lat,indlat,lon,indlon):
    loclat=[]
    loclon=[]
    for i in range(len(indlat)):
        loclat.append(np.where(lat>=indlat[i])[0][0])
        loclon.append(np.where(lon<=indlon[i])[0][0])
    return loclat,loclon


# In[7]:


ind_lon_cities = [ -115.939167, -116.605833, -117.1625]
ind_lat_cities = [ 30.556389, 31.857778, 32.715]

ind_lat,ind_lon=closest(lat_acVEL,ind_lat_cities,lon_acVEL,ind_lon_cities)


# In[8]:


matfile=loadmat( '../datafiles/PALL.mat')
dep=matfile['d'][0][:11]
latout=matfile['lat'][0][:11]
lonout=matfile['lon'][0][:11]


# In[9]:


#Only for at depth smooth!! 

latloc=[]
lonloc=[]

for i in range(len(dep)):
    ind=np.where(dep[i][0]>=500)[0][0]
    latloc.append(latout[i][0][ind])
    lonloc.append(lonout[i][0][ind])

loclatIn,loclonIn=closest(lat_acVEL,latloc,lon_acVEL,lonloc)


#Only for at depth Org

latlocorg=[]
lonlocorg=[]

for i in range(len(dep)):
    ind=np.where(dep[i][0]>=500)[0][0]
    latlocorg.append(latout[i][0][ind])
    lonlocorg.append(lonout[i][0][ind])

loclatInorg,loclonInorg=closest(lat_acVELo,latlocorg,lon_acVELo,lonlocorg)


# In[10]:


end=np.where(lat_acVEL>33.5)[0][0]
endOrg=np.where(lat_acVELo>33.5)[0][0]


# In[11]:


params = {'font.size': 10,
          'figure.figsize': (7.48, 6.732),
         'font.family':'sans'}
pl.rcParams.update(params)
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000


# In[12]:


fig = plt.figure()
gs = GridSpec(nrows=2, ncols=5, height_ratios=[1, 0.8])

const=1e6
vmin=0
vmax=2

title='PSD [10$^{-6}$ (ms$^{-1}$)$^{2}$Hz$^{-1}$]'
xlab='Frequency [cpd]'
ylab='Distance [km]'

ax = fig.add_subplot(gs[0, 0:2])


ax.text(-0.08, 1.02, '(a)', fontweight='bold', color='k', 
        transform=ax.transAxes)

cax = ax.pcolormesh(freqfilt[1,:]*(24*3600),distVEL[:end],psdfilt[:end,:]*const,vmin=vmin,vmax=vmax, cmap=cmocean.cm.amp)

divider = make_axes_locatable(ax)
axdiv = divider.new_vertical(size = '5%', pad = 0.4)
fig.add_axes(axdiv)
cbar_ax = plt.colorbar(cax, cax=axdiv,orientation='horizontal')
cbar_ax.ax.xaxis.set_label_position("top")
cbar_ax.set_label(title)
ax.set_xlim((0,0.000016*(24*3600)))


ax.set(ylabel=ylab)
ax.set_title('Smooth')


for ll, lab in zip(ind_lat,
                       ['San Quintín', 'Ensenada', 'San Diego']):
    ax.plot(0, distVEL[ll], "_",markersize=50, color='black',zorder=5)
    ax.text(0, distVEL[ll]+5, lab, fontsize=9,color='black')

ax1 = fig.add_subplot(gs[1, 0:2])
ax1.set(xlabel=xlab, ylabel=ylab)
ax1.text(-0.08, 1.05, '(b)', fontweight='bold', color='k', 
        transform=ax1.transAxes)

cax1 = ax1.pcolormesh(freqfilto[1,:]*(24*3600),distVELo[:endOrg],psdfilto[:endOrg,:]*const,vmin=vmin,vmax=vmax, cmap=cmocean.cm.amp)

ax1.set_xlim((0,0.000016*(24*3600)))

ax1.set(xlabel=xlab, ylabel=ylab)

omega=(2*np.pi)/(23*60*60 + 56*60 + 4.1)


ax1.set_title('Realistic')

axin = fig.add_subplot(gs[:,2:])

shift=1.8
for nr in np.arange(0,len(loclatIn),1):
    axin.axhline(shift*nr,color='red',zorder=1,linewidth=1)
    if nr == 0:
        axin.plot((freqfilt[1])*(24*3600),(psdfilt[loclatIn[nr]]*const)+shift*nr,c='k',linewidth=1.5,zorder=20,label='Smooth') 
        axin.plot((freqfilto[1])*(24*3600),(psdfilto[loclatInorg[nr]]*const)+shift*nr,alpha=0.6,c='k',linewidth=1.5,linestyle='dashed',zorder=10,label='Realistic') 
        axin.scatter((2*omega*np.sin(np.deg2rad(lat_acVEL[loclatIn[nr]]))*24*3600)/(2*np.pi),shift*nr,color='blue',marker="|",s=200,label='Inertial Frequency')
    else:
        axin.plot((freqfilt[1])*(24*3600),(psdfilt[loclatIn[nr]]*const)+shift*nr,c='k',linewidth=1.5,zorder=20) 
        axin.scatter((2*omega*np.sin(np.deg2rad(lat_acVEL[loclatIn[nr]]))*24*3600)/(2*np.pi),shift*nr,color='blue',marker="|",s=200)
        axin.plot((freqfilto[1])*(24*3600),(psdfilto[loclatInorg[nr]]*const)+shift*nr,alpha=0.6,c='k',linewidth=1.5,linestyle='dashed',zorder=10) 
    ax1.axhline(distVEL[loclatIn[nr]],color='red',linestyle='dashed',linewidth=1,alpha=0.7)
    ax.axhline(distVEL[loclatIn[nr]],color='red',linestyle='dashed',linewidth=1,alpha=0.7)

for ll, lab in zip(ind_lat,
                       ['San Quintín', 'Ensenada', 'San Diego']):
    ax1.plot(0, distVEL[ll], "_",markersize=50, color='black',zorder=5)
    ax1.text(0, distVEL[ll]+5, lab, fontsize=9,color='black')

axin.axvline((freqfilt[1][np.argmax(psdfilt[loclatIn[nr]])])*(24*3600),linestyle='dotted',color='k',linewidth=1.5)
axin.legend(loc='upper left',fontsize=8)
xlab='Frequency [cpd]'

axin.set_xlim((0,0.000016*(24*3600)))

axin.set(xlabel=xlab, ylabel=title)
axin.set_xticks(freqfilt[1][:15]*(24*3600))

axin.minorticks_on()

axin.grid(which='minor',linestyle='--', alpha=0.5)
axin.grid(which='major')
axin.set_axisbelow(True)

axin.text(-0.17, 1, '(c)', fontweight='bold', color='k', 
        transform=axin.transAxes)

fig.tight_layout()
plt.savefig('../figures/PSD_fig5.pdf',bbox_inches='tight')






