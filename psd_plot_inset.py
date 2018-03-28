#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function,division

import os
import sys
import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pyfits
#import triangle
import fnmatch
import gatspy.periodic as gp
from scipy.optimize import curve_fit

def med_filt(x, y, dt):
    """
    De-trend a light curve using a windowed median.
    """
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    assert len(x) == len(y)
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = (x >= t - 0.5 * dt) * (x <= t + 0.5 * dt)
        r[i] = np.nanmedian(y[inds])
    return r

def model_func(params,f,H=1):
    a1,a2,b1,b2,c1,c2,W,height,sigma,numax,a3,b3,c3 =params
    gran1=(a1**2/b1)/(1+((f/b1)**c1))
    gran2=(a2**2/b2)/(1+((f/b2)**c2))
    act=(a3**2/b3)/(1+((f/b3)**c3))
    osc=H*height*(np.exp(-((f-numax)**2)/(2*(sigma**2))))
    vnyq=f[-1]
    model= (act+osc+gran1+gran2)*(np.sinc(0.5*(f/vnyq)))**2 + W
    return model

def rebin(x, r):
    m = len(x) // r
    return x[:m*r].reshape((m,r)).mean(axis=1),r

def split(delimiters, string, maxsplit=0):
    import re
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def psps(freq,power,idx,N=1):

    freq=freq/1e6#-->Hz
    nyq=1/(2*(freq[1]-freq[0]))
    df=1/freq[-1]

    t,psp=gp.lomb_scargle_fast.lomb_scargle_fast(freq[idx],power[idx],f0=df,df=df/N,Nf=N*(nyq/df))
    return t,psp

def gaus2(x,a,x0,sigma,c):
    return (a*np.exp(-(x-x0)**2/(2*sigma**2)))**2+c

def gausss(x,a1,x1,sigma1,a2,x2,sigma2,c):
    return (a2*np.exp(-(x-x2)**2/(2*sigma2**2)))**2+ (a1*np.exp(-(x-x1)**2/(2*sigma1**2)))**2 +c


############################PSD########################
i='/home/thomas/Dropbox/PhD/Year_3/Year_3_koi6194_kic9145861/PSD/kplr009145861_kasoc-psd_llc_v1.fits'
mypath='/home/thomas/Dropbox/PhD/Year_3/Year_3_koi6194_kic9145861/MCMC_Results/'

#i='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C10-EPIC_228737206/KASOC_DATA/ktwo228737206_kasoc-psd_slc_v1.fits'
#mypath='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C10-EPIC_228737206/PSD'

#i='/home/thomas/Dropbox/PhD/Written Papers/KOI-3890_Manuscript/Old/scripts/PDCspec8564976.pow'
#mypath='/home/thomas/Dropbox/PhD/Written Papers/KOI-3890_Manuscript/New/Tom/PSD_results'

#i='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C5_EPIC_211403356/KASOC/ktwo211403356_kasoc-psd_llc_v1.fits'
#mypath='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C5_EPIC_211403356/KASOC/PSD'

#i='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C8-EPIC_220548055_bad_supp_l1/ktwo220548055_01_kasoc-psd_llc_v1.fits'
#mypath='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C8-EPIC_220548055_bad_supp_l1/PSD'

#i='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C10-EPIC_228737206/KASOC_DATA/ktwo228737206_kasoc-psd_slc_v1.fits'
#mypath='/home/thomas/Dropbox/PhD/Year_2/K2_radial_hosts/C10-EPIC_228737206/PSD'

#i='/home/thomas/Dropbox/PhD/Year_3/Other_planet_hosts/C5_EPIC_211403356/K2P2_EPIC211403356_20161107_171847-20161107-171919/ktwo211403356_01_kasoc-psd_llc_v1.pow'
#mypath='/home/thomas/Dropbox/PhD/Year_3/Other_planet_hosts/C5_EPIC_211403356/BG/'

#i='/home/thomas/Dropbox/PhD/Year_3/Any_other_Retired?/C7_EPIC_215847238/C7_EPIC_215847238_ps_llc.txt'
#mypath='/home/thomas/Dropbox/PhD/Year_3/Any_other_Retired?/C7_EPIC_215847238/C7_EPIC_215847238/PSD'

#i='/home/thomas/Dropbox/PhD/Year_3/Any_other_Retired?/C10_EPIC_228739714/c10_EPIC_228739714_ps_llc.txt'
#mypath='/home/thomas/Dropbox/PhD/Year_3/Any_other_Retired?/C10_EPIC_228739714/PSD'

#i='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C5_211351816/ktwo211351816_kasoc-psd_llc_v1.fits'
#mypath='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C5_211351816/PSD'

#i='/home/thomas/Dropbox/PhD/Python Codes/kplr006442183_kasoc-psd_slc_v1.fits'
#mypath='/home/thomas/Dropbox/PhD/Python Codes/KIC_6442183'

#############################################################
#Sam
#i='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C10_228754001/228754001.dat.ts.fft'
#mypath='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C10_228754001/PSD'

#i='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C5_211351816/211351816.dat.ts.fft'
#mypath='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C5_211351816/PSD'

#i='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C1_201132839/201132839.dat.ts.fft'
#mypath='/home/thomas/Dropbox/PhD/Year_3/RG_hosts_sam_g/C1_201132839/PSD'


q1q16=np.loadtxt('/home/thomas/Dropbox/PhD/Year_3/revisedq1q16.txt',usecols=(0,1,2),skiprows=49)

############################PSD########################
if i.endswith('.fits'):
	FITSfile=pyfits.open(i)
	topheader=FITSfile[0].header
	dataheader=FITSfile[1].header
	kic=topheader['keplerid']
	data=FITSfile[1].data
	f,psd=data['frequency'],data['psd']

elif i.endswith('.txt') or i.endswith('.dat') or i.endswith('.pow') or i.endswith('.fft'):
	f,psd=np.loadtxt(i,unpack=True)
	psd=psd[f<283]
	f=f[f<283]
	#psd=psd*1e6
	#psd=psd[:len(psd)//2]
	#f=f[:len(f)//2]
	
	kic=None

else: 
	print('No')
	sys.exit()


psd_ori=psd#copy of psd for later


if os.path.isfile(mypath+'/results.txt'): 

	res=np.loadtxt(mypath+'/results.txt')

	'''
	bw=f[1]-f[0]
	nbins=int(0.5/bw)
	print(nbins)
	psd=rebin(psd,nbins)[0]
	f=rebin(f,nbins)[0]
	'''

	params=res[:,1]
	nu=res[-4]
	enumax=np.mean([nu[2]-nu[1],nu[1]-nu[0]])
	psd=psd/model_func(params,f,H=0)
	numax=params[-4]

else:
	print('median')
	med=1.42*med_filt(f,psd,dt=25.)
	psd=psd/med
	numax=float(f[psd==max(psd)])
	enumax=np.nan

plt.plot(f,psd,'k')
plt.show()

print('PSPS,for delnu')


delnu_in=0.276*numax**0.751
denv=numax/2

idx=np.where((f>(numax-denv)) & (f<(numax+denv)))[0]

plt.plot(f[idx],psd[idx],'k',alpha=0.6)
plt.show()


dt,psp=psps(f,psd,idx,N=10)

delnu_scale=(1e6/dt)*2#doubled

plt.figure()
plt.plot(delnu_scale,psp,'k')

dt,psp=psps(f,psd,idx,N=1)
delnu_scale=(1e6/dt)*2#doubled
plt.plot(delnu_scale,psp,'r')

plt.axvline(x=delnu_in,ls='--',c='b',lw=3,label='delnu scaling relation')
plt.legend(loc='best')
plt.show()


delnu=delnu_scale[psp==max(psp[(delnu_scale<1.4*delnu_in) & (delnu_scale>delnu_in/1.4)])]

print('Numax,enumax',numax,enumax)
print('Delnu(numax)',delnu_in)
print('Delnu(MAX PSPS near delnu(numax))',delnu)


idx=np.where((delnu_scale<1.5*delnu) & (delnu_scale>delnu/1.5))[0]

#popt,pcov = curve_fit(gausss,delnu_scale[idx],psp[idx],p0=[max(psp),delnu,0.25,max(psp)/4,delnu+2,0.25,0],maxfev=10000)
#amp1,delnu,edelnu,amp2,d02,ed02,bg=popt.T

popt,pcov = curve_fit(gaus2,delnu_scale[idx],psp[idx],p0=[max(psp),delnu,0.25,0],maxfev=10000)
print(pcov)
amp1,delnu,edelnu,bg=popt.T

print('Delnu fitted',delnu,edelnu)

plt.plot(delnu_scale[idx],psp[idx])
plt.axvline(x=delnu,ls='--',c='g',lw=3)
plt.axvline(x=delnu_in,ls='--',c='r',lw=3)
#plt.plot(delnu_scale[idx],gausss(delnu_scale[idx],*popt),'r:',label='fit')
plt.plot(delnu_scale[idx],gaus2(delnu_scale[idx],*popt),'r:',label='fit')
plt.show()



####################################################################
####################################################################
###################Nice PS and inset PSPS plot#####################

idx=np.where((f>(numax-denv)) & (f<(numax+denv)))[0]

plt.plot(f,psd_ori,'k',alpha=0.4,lw=1)
plt.plot(f[idx],psd_ori[idx],'k',lw=1)
plt.plot(f,model_func(params,f),'r',lw=2)
plt.plot(f,model_func(params,f,H=0),'r--',lw=2)
plt.axvspan(xmin=numax-enumax,xmax=numax+enumax,color='b',alpha=0.1)
plt.axvline(x=numax,c='b',ls='--')
plt.yscale('log')
plt.xlabel('Frequency ($\mu$Hz)', fontsize=14)
plt.ylabel('Power Spectral Density (ppm$^2\mu$Hz$^{-1}$)', fontsize=14)
plt.xlim(20,280)
plt.ylim(1e1,1e5)

dt,psp=psps(f,psd,idx,N=5)
delnu_scale=(1e6/dt)*2#doubled

a = plt.axes([.575, .575, .3, .3])
plt.plot(delnu_scale,psp,'k')
plt.axvline(x=delnu_in,ls='-',c='r',lw=1)
plt.axvline(x=delnu,ls='--',c='r',lw=1)
plt.axvspan(xmin=delnu-edelnu,xmax=delnu+edelnu,alpha=0.2,color='r')
plt.xlabel(r'$\Delta\nu$ ($\mu$Hz)',labelpad=-3)
plt.ylabel(r'PSPS',labelpad=0)
plt.xlim(0,15)
plt.ylim(0,0.02)
plt.yticks([])

plt.savefig('/home/thomas/Dropbox/PhD/Written Papers/Exoplanet_Host_KOI_6194/psd_psps.pdf')
plt.show() 

