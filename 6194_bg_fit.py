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
from os.path import expanduser
from tqdm import tqdm
from time import sleep

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



def initial(numax,f,psd):
	a1=3382*numax**-0.609
	a2=3382*numax**-0.609
	b1=0.317*(numax**0.97)
	b2=0.948*(numax**0.992)
	c1,c2=4,4

	a3=0.1*a2
	b3=0.1*b1
	c3=2.5
	
	#cc=1.28*10**(0.4*(12.0-kepmag)+7.0)#shot noise
	#sigma_shot=(1e6/cc)*(np.sqrt(cc+9.5e5*(14.0/kepmag)**5))
	P_shot=np.mean(psd[f>f[-10]]) #mean of last 100 bins
	#P_shot=#(2e-6*58.85*sigma_shot**2)
	height=2.03e7*(numax**-2.38)
	if numax>200:
		delta_env=numax/2
	else:
		delta_env= 0.66*(numax**0.88)
	sigma=delta_env*(1/(2*np.sqrt(2*np.log(2))))

	return [a1,a2,b1,b2,c1,c2,P_shot,height,sigma,numax,a3,b3,c3]


def model_func(params,f):
	a1,a2,b1,b2,c1,c2,W,height,sigma,numax,a3,b3,c3 =params
	gran1=(a1**2/b1)/(1+((f/b1)**c1))
	gran2=(a2**2/b2)/(1+((f/b2)**c2))
	act=(a3**2/b3)/(1+((f/b3)**c3))
	osc=height*(np.exp(-((f-numax)**2)/(2*(sigma**2))))
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



#q1q16=np.loadtxt('/home/thomas/Dropbox/PhD/Year_3/revisedq1q16.txt',usecols=(0,1,2),skiprows=49)
home=expanduser('~')

############################PSD########################
file=home+'/Dropbox/PhD/Year_4/Year_3_koi6194_kic9145861/kplr009145861_kasoc-psd_llc_v1.fits'
loc=home+'/Dropbox/PhD/Year_4/Year_3_koi6194_kic9145861/BG_FIT/'

############################PSD########################
if file.endswith('.fits'):
	FITSfile=pyfits.open(file)
	topheader=FITSfile[0].header
	dataheader=FITSfile[1].header
	kic=topheader['keplerid']
	data=FITSfile[1].data
	f,psd=data['frequency'],data['psd']

elif file.endswith('.txt') or file.endswith('.dat') or file.endswith('.pow') or file.endswith('.fft'):
	f,psd=np.loadtxt(file,unpack=True)
	psd=psd[f<283]
	f=f[f<283]
	#psd=psd*1e6
	#psd=psd[:len(psd)//2]
	#f=f[:len(f)//2]
	
	kic=None

else: 
	print('No')
	sys.exit()

bw=f[1]-f[0]


###########################################
#Opened file and cuts etc, now do quick max L check for initial guess
if f[-1]<300:
	nu=np.arange(1,f[-1],1)
else:
	nu=np.linspace(1,f[-1],500)
	
m=[]
for n in tqdm(nu):
	mod=initial(n,f,psd)
	model=model_func(mod,f)
	chi_sqr = -1.0*np.sum(np.log(model)+(psd/model))#
	m.append(chi_sqr)

numax=nu[m==max(m)]
mod=initial(numax,f,psd)
model=model_func(mod,f)
print('max chi sqr',max(m))
print('numax',numax)

###########################################
if numax<100:
	nbins=int(0.2/bw)
elif numax<150:
	nbins=int(1/bw)
elif numax<500:
	nbins=int(2/bw)
	
#elif numax>1000:
#	nbins=int(15/bw)

else:
	nbins=int(10/bw)

if nbins<1:
	nbins=1



if f[-1]>300:#slc probably
	psd=psd[f>10]
	f=f[f>10]
	#Cheat for subgiants
	psd=psd[f<numax*2]
	f=f[f<numax*2]


else:#long cadence
	psd=psd[f>4]
	f=f[f>4]	




f_ori,psd_ori=f,psd#hold for later

f=rebin(f,nbins)[0]
psd=rebin(psd,nbins)[0]	

guess=initial(numax,f,psd)#a1,a2,b1,b2,c1,c2,W,height,sigma,numax
guess=np.array(guess)
labels=['a1','a2','b1','b2','c1','c2','W','H','sig','numax','a3','b3','c3']

print('KIC:',str(kic))
print(np.c_[labels,guess])

if not os.path.isdir(loc):
   os.makedirs(loc)

np.savetxt(loc+'/initial.txt',guess,header=str(kic) + '\n' +str(labels))
plt.figure()
plt.plot(f,model_func(guess,f),'r',label='Initial Guess')
plt.plot(f,psd,'k',alpha=0.6,label='PSD')	
plt.axvline(guess[-2],c='b',ls='--')
plt.xlim(f.min(),f.max())
plt.xlabel(r"$\nu$")
#plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig(loc+"/initial.png")
plt.show()






########################################
########################################
def lnprior(theta,guess):
	snumax=guess[-4]
	a1,a2,b1,b2,c1,c2,W,height,sigma,numax,a3,b3,c3 = theta
	if snumax<100:
		if 0.0 <= a1 < 4*guess[0] and 0.0 <= a2 < 4*guess[1] and 0.25*guess[2] <= b1 < 4*guess[2] and 0.25*guess[3] <= b2 < 4*guess[3]  \
		and 2.0 <= c1 < 6.0 and 2.0 <= c2 < 6.0 and 0.0 <= W < 1e6 and 0<=height<=1e6 and (snumax/100)<=sigma<(snumax/2) \
		and 0<=numax<3.0*snumax and 0<=a3<=1e9 and 0<=b3<=b1 and 0<=c3<=5.0:
			return 0.0


	else:
		if 0.0 <= a1 < 4*guess[0] and 0.0 <= a2 < 4*guess[1] and 0.5*guess[2] <= b1 < 2*guess[2] and 0.5*guess[3] <= b2 < 2*guess[3]  \
		and 2.0 <= c1 < 6.0 and 2.0 <= c2 < 6.0 and 0.0 <= W < 1e6 and 0<=height<=1e6 and (snumax/100)<=sigma<(snumax/2) \
		and 0.5*snumax<=numax<2.0*snumax and 0<=a3<=1e9 and 0<=b3<=b1 and 0<=c3<=6.0:
			return 0.0

	return -np.inf

def lnlike(theta,f,psd):
	a1,a2,b1,b2,c1,c2,W,height,sigma,numax,a3,b3,c3 = theta
	model =model_func(theta,f)#rebins
	chi_sqr = -1.0*nbins*np.sum(np.log(model)+(psd/model))#

	return chi_sqr

def lnprob(theta, f,psd):
	lp = lnprior(theta,guess)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta,f,psd)


# Set up the sampler.

nwalkers, niter, ndim = 100, 2000, int(len(guess))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(f,psd))

p0 = np.zeros([nwalkers, ndim])
for j in range(nwalkers):
	p0[j,:] = guess + 1e-2*np.random.randn(ndim)
	
print('... burning in ...')
for p, lnprob, state in tqdm(sampler.sample(p0, iterations=niter),total=niter):
	sleep(0.01)


# Clear and run the production chain.
sampler.reset()
print('... running sampler ...')
for p, lnprob, state in tqdm(sampler.sample(p, lnprob0=lnprob,iterations=niter),total=niter):	
	sleep(0.001)


#sample=sampler.chain[:, :, :]	#all chains and samples
#np.save(loc+'/sample_out',sample)

fig, axes = plt.subplots(len(guess), 1, sharex=True, figsize=(8, 9))

for i in range(0,len(guess)):
	axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
	#axes[i].axhline(m_true, color="#888888", lw=2)
	axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
fig.savefig(loc+"/chain.png")
plt.close(fig)

#print('n temps:', ntemps, "log evidence: ", sampler.thermodynamic_integration_log_evidence())

# Make the corner plot.
burnin = 0#int(niter/2)
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) #burnt and flattened chains

quantiles = np.percentile(samples,[16,50,84],axis=0).T 
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
for i in range(0,len(labels)):
	print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

np.savetxt(loc+'/results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')

sampler.reset()

##########################################
fig = corner.corner(samples, labels=labels)
fig.savefig(loc+"/corner.png")
plt.close(fig)

# Plot some samples onto the data.
plt.figure()
plt.plot(f,psd ,'k-',alpha=0.2)
for theta in samples[np.random.randint(len(samples), size=int(niter/25))]:
#    a1,a2,b1,b2,c1,c2,W,height,sigma,numax, a3,c3 = 
	plt.plot(f, model_func(theta,f), 'b-', alpha=0.1)

plt.plot(f, model_func(medians,f), 'r-', alpha=0.8)
plt.xlim(f.min(),f.max())
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\nu$")
plt.ylabel("PSD")
plt.tight_layout()
plt.savefig(loc+"/fits.png")
plt.close()


plt.figure()
plt.plot(f,psd ,'k-',alpha=0.4)

params[7]=0
plt.plot(f, model_func(medians,f), 'r-', alpha=0.4)
plt.xlim(f.min(),f.max())
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\nu$")
plt.ylabel("PSD")
plt.tight_layout()
plt.savefig(loc+"/fits_bg.png")

plt.clf()
medians[7]=0#H=0
plt.plot(f, psd/model_func(medians,f), 'r-', alpha=0.4)
plt.xlim(f.min(),f.max())
plt.xlabel(r"$\nu$")
plt.ylabel("PSD")
plt.tight_layout()
plt.savefig(loc+"/div_by.png")
plt.close()

print('Done \n')

###################################################
###################################################
###################################################

print('PSPS,for delnu')
f,psd=f_ori,psd_ori#recast to original size etc


#med=1.42*med_filt(f,psd,dt=25.)
#psd=psd/med
psd=psd/model_func(medians,f)


numax=params[-4]
delnu_in=0.276*numax**0.751
denv=numax/2

idx=np.where((f>(numax-denv)) & (f<(numax+denv)))[0]
plt.plot(f[idx],psd[idx],'k',alpha=0.6)
plt.show()


dt,psp=psps(f,psd,idx,N=5)
delnu_scale=(1e6/dt)*2#doubled

plt.figure()
plt.plot(delnu_scale,psp)
plt.axvline(x=delnu_in,ls='--',c='b',lw=3,label='delnu scaling relation')
plt.legend(loc='best')
plt.show()

print('Delnu scaling',delnu_in)
delnu=delnu_scale[psp==max(psp[(delnu_scale<1.4*delnu_in) & (delnu_scale>delnu_in/1.4)])]
print('Delnu max psps',delnu)


idx=np.where((delnu_scale<1.5*delnu) & (delnu_scale>delnu/1.5))[0]
#popt,pcov = curve_fit(gausss,delnu_scale[idx],psp[idx],p0=[max(psp),delnu,0.25,max(psp)/4,delnu+2,0.25,0],maxfev=10000)
#amp1,delnu,edelnu,amp2,d02,ed02,bg=popt.T

popt,pcov = curve_fit(gaus2,delnu_scale[idx],psp[idx],p0=[max(psp),delnu,0.25,0],maxfev=10000)
amp1,delnu,edelnu,bg=popt.T


print('Delnu fit',delnu)
print('delnu,edelnu',delnu,edelnu)

np.savetxt(loc+'/Delnu.txt',np.array((delnu,edelnu)),header='Delnu\tedlenu',delimiter='\t')

plt.plot(delnu_scale[idx],psp[idx])
plt.axvline(x=delnu,ls='--',c='g',lw=3)
plt.axvline(x=delnu_in,ls='--',c='r',lw=3)
plt.plot(delnu_scale[idx],gaus2(delnu_scale[idx],*popt),'r:',label='fit')
plt.show()




