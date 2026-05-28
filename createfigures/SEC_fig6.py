#!/usr/bin/env python
# coding: utf-8

# In[15]:


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
import pylab as pl
import cmocean
from os.path import exists
from scipy.io import loadmat


# In[16]:


corrinds=[29.96,30.34,30.69,31.03,31.36,31.69,32.05,32.39,32.77,33.15,33.44]


# In[17]:


coast='smooth'
ds=xr.open_dataset('../datafiles/' + str(coast)+'/ashore.nc')


# In[18]:


day1=ds.TIME[0].values[108]/(60*60*24)
day2=ds.TIME[0].values[144]/(60*60*24)
day3=ds.TIME[0].values[180]/(60*60*24)
ik=7
X=ds.X[ik].values
Z=ds.Z[ik].values
VALfilt=ds.Filtashore[ik].values*100
TIME=ds.TIME[ik].values

latUsed=corrinds[ik]


# In[28]:


WR=[]
RRR=[]
for l in np.arange(0,4,1):
    name='../datafiles/ctwmodel/dispr' + str(latUsed) + 'mode' +str(l) + '.mat'
    if exists(name) == True:
        dispfile=loadmat(name)
        wr, rrr = dispfile['dispcurve'][:,0],dispfile['dispcurve'][:,1]
        WR.append(wr)
        RRR.append(rrr)

matfile=loadmat( '../datafiles/ctwmodel/freqs.mat')
freqs=matfile['freq'][0]


# ## Plotting modes vs MITgcm output

# In[20]:


def get_Brink(file_fig):#,file_h): #, file_ratio):
    # Brink mode
    file = loadmat(file_fig)
    z, xpl, xxx, zzz, xgr, zgr = file['z'][0,:], file['xpl'][0,:], file['xxx'][0,:], file['zzz'][0,:], file['xgr'], file['zgr']
    k, omega,epe,eke = file['wavenumber'][0][0], file['frequency'][0][0], file['epe'][0][0],file['eke'][0][0]

    # (u is cross-shore and v is alongshore in Brink.)
    p0, u0, v0, w0, r0 = file['p'], file['u'],file['v'], file['wvel'], file['rho']

    scale=0.2
    w = w0.transpose() * 0.01 * scale # cms-1 to ms-1 and normalization (?)
    u = u0.transpose() * 0.01 * scale # cms-1 to ms-1 and normalization 
    v = v0.transpose() * 0.01 * scale # cms-1 to ms-1 and normalization 
    r = r0.transpose() * 1.0 * scale # mg/cm³ to kg/m³ and normalization
    p = p0.transpose() * 0.1 * scale # dyn/cm² to 0.1 Pa (or kg m-1 s-2) and normalization

    return(u,v,w,r,p,z,k,omega,xpl, xxx, zzz,zgr.transpose(),xgr.transpose(), epe, eke)

def openbrink(loc,coast):
	u=[]
	v=[]
	w=[]
	r=[]
	p=[]
	k=[]
	omega=[]
	epe=[]
	eke=[]


	for l in np.arange(0,5,1):
		if exists( '../datafiles/ctwmodel/dataSVB'+ str(loc) +'mode' + str(l) + '.mat') == True:
			uo,vo,wo,ro,po,z,ko,omegao, xpl, xxx, zzz, zgr, xgr, epeo, ekeo = get_Brink( '../datafiles/ctwmodel/dataSVB'+ str(loc) +'mode' + str(l) + '.mat')	
		else: 
			uo=0
			vo=0
			wo=0
			ro=0
			po=0
			ko=0
			omegao=0
			epeo=0
			ekeo=0

		u.append(uo) 
		v.append(vo)
		w.append(wo)
		r.append(ro)
		p.append(po)
		k=np.append(k,ko)
		omega=np.append(omega,omegao)
		epe=np.append(epe,epeo)
		eke=np.append(eke,ekeo)

	return u,v,w,r,p,k,omega,epe,eke,xgr,zgr

def plotBrink(ax,grid_X,grid_Z,levelsb,xlab,ylab,modenr,nr,varbrink,lat,t,colormap):
    ax.set_facecolor('tan')
    cax=ax.contourf(grid_X,grid_Z,varbrink ,cmap=colormap,levels=levelsb)
    ax.contour(grid_X,grid_Z,varbrink , levels=[0], linewidths=2, 
                linestyles='-', colors='k', zorder=2)
    ax.xaxis.set_inverted(True)
    if nr==-1:
        ax.set(xlabel=xlab)
        if t>2.5:
            ax.tick_params(axis='y',which='both', left=True, right=False, labelleft=False)
        else:
            ax.set(ylabel=ylab)          
    else:
        ax.set_title(f'CTW-model \nMode nr {modenr}')
        ax.set(xlabel=xlab)
        if modenr>1:
            ax.tick_params(axis='y',which='both', left=True, right=False, labelleft=False)
        else:
            ax.set(ylabel=ylab)
    return cax


# In[21]:


topp=[184,276,349,427,508,582]
dal=[104,228,312,387,469,540]


# In[22]:


params = {'font.size': 9,
          'figure.figsize': (7.48, 4.4),
         'font.family':'sans'}
pl.rcParams.update(params)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# In[29]:


vmin=-np.nanmax(abs(VALfilt))*1e2
vmax=np.nanmax(abs(VALfilt))*1e2

levels=np.linspace(vmin,vmax,15)

u,v,w,r,p,k,omega,epe,eke,xgr,zgr = openbrink(latUsed,coast)

indZ=np.where(Z<np.min(zgr))[0][0]


valsin=v
clabel='Alongshore velocity\n[10$^{-2}$ cms$^{-1}$]' 
clabelB='Alongshore velocity [10$^{-9}$ cms$^{-1}$]' 
colormap=cmocean.cm.balance

modes=0
modenr=[]
vals=[]
for i in range(len(valsin)):
    if np.any(valsin[i]!=0):
        modenr.append(i)
        modes=modes+1
        vals.append(valsin[i])


vals=np.array(vals)

vminb=-np.nanmax(abs(vals))*1e9
vmaxb=np.nanmax(abs(vals))*1e9

levelsb=np.linspace(vminb,vmaxb,15)

xlab='Cross-shore distance [km]'
ylab='Depth [m]'


fig = plt.figure()
if len(vals)==3:
    gs = GridSpec(nrows=2, ncols=9,wspace=0.5,hspace=0.7)
elif len(vals)==2:
    gs = GridSpec(nrows=2, ncols=2)

ax = fig.add_subplot(gs[0, 1:4])
ax.text(0.93, 1.06, '(a)', fontweight='bold', color='k', 
transform=ax.transAxes)

cax2=plotBrink(ax,X,Z[:indZ],levels,xlab,ylab,-1,-1,np.mean(VALfilt[dal,:indZ,:],axis=0)*1e2,latUsed,day1,colormap)
ax.set_ylim((-1000,0))
ax.set_title(f'MITgcm\nAverage trough')

ax = fig.add_subplot(gs[0, 4:7])
ax.text(0.93, 1.06, '(b)', fontweight='bold', color='k', 
transform=ax.transAxes)


cax2=plotBrink(ax,X,Z[:indZ],levels,xlab,ylab,-1,-1,np.mean(VALfilt[topp,:indZ,:],axis=0)*1e2,latUsed,day3,colormap)
ax.set_ylim((-1000,0))
ax.set_title(f'MITgcm\nAverage peak')

cbar_ax = fig.add_axes([0.74, 0.61, 0.01, 0.26])        
fig.colorbar(cax2, cax=cbar_ax)
cbar_ax.set_ylabel(clabel)
cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax = fig.add_subplot(gs[1, 0:3])
ax.text(0.93, 1.06, '(c)', fontweight='bold', color='k', 
transform=ax.transAxes)

cax2=plotBrink(ax,xgr,zgr,levelsb,xlab,ylab,1,1,vals[0]*1e9,latUsed,day1,colormap)
ax.set_ylim((-1000,0))

fp=np.where((RRR[0]/(2*np.pi))*3600*24<freqs[ik][0][0]*3600*24)[0][-1]
c=(RRR[0][fp]/(2*np.pi))/(WR[0][fp]*100/(2*np.pi))
ax.text(0.74, 0.01, f' Phase\n speed:\n{c:.1f} m/s', color='k', 
transform=ax.transAxes, fontsize=9)

ax = fig.add_subplot(gs[1, 3:6])
ax.text(0.93, 1.06, '(d)', fontweight='bold', color='k', 
transform=ax.transAxes)

cax2=plotBrink(ax,xgr,zgr,levelsb,xlab,ylab,2,2,vals[1]*1e9,latUsed,day1,colormap)
ax.set_ylim((-1000,0))

fp=np.where((RRR[1]/(2*np.pi))*3600*24<freqs[ik][0][0]*3600*24)[0][-1]
c=(RRR[1][fp]/(2*np.pi))/(WR[1][fp]*100/(2*np.pi))
ax.text(0.74, 0.01, f' Phase\n speed:\n{c:.1f} m/s', color='k', 
transform=ax.transAxes, fontsize=9)



if len(vals)==3:
    ax = fig.add_subplot(gs[1, 6:9])
    ax.text(0.93, 1.06, '(e)', fontweight='bold', color='k', 
    transform=ax.transAxes)

    cax2=plotBrink(ax,xgr,zgr,levelsb,xlab,ylab,3,3,vals[2]*1e9,latUsed,day1,colormap)
    ax.set_ylim((-1000,0))

    fp=np.where((RRR[2]/(2*np.pi))*3600*24<freqs[ik][0][0]*3600*24)[0][-1]
    c=(RRR[2][fp]/(2*np.pi))/(WR[2][fp]*100/(2*np.pi))
    ax.text(0.74, 0.01, f' Phase\n speed:\n{c:.1f} m/s', color='k', 
    transform=ax.transAxes, fontsize=9)

cbar_ax = fig.add_axes([0.91, 0.125, 0.01, 0.26])        
fig.colorbar(cax2, cax=cbar_ax)
cbar_ax.set_ylabel(clabelB)
cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.suptitle('At latitude ' + str(latUsed) + '°N        ',y=1.01)

plt.savefig('../figures/SEC_fig6.pdf')


