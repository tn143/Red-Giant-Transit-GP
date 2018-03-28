#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function,division
import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner
from os.path import expanduser
import sys
from scipy import stats
import batman
import pandas as pd
import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy import optimize
from tqdm import tqdm
import os
from scipy import constants
import celerite
from celerite import terms
from celerite.modeling import Model, ConstantModel
import pyfits

G=constants.G
home=expanduser('~')
#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))
params = batman.TransitParams()
#########PSD MODEL############
def model_func(params,f):
	a1,a2,b1,b2,W,height,sigma,numax,a3,b3 =params
	gran1=(a1**2/b1)/(1+((f/b1)**4))
	gran2=(a2**2/b2)/(1+((f/b2)**4))
	act=(a3**2/b3)/(1+((f/b3)**4))
	osc=height*(np.exp(-((f-numax)**2)/(2*(sigma**2))))
	vnyq=f[-1]
	model= (act+osc+gran1+gran2)*(np.sinc(0.5*(f/vnyq)))**2 + W
	return model
#########PSD MODEL############

def fold(time, period, origo=0.0, shift=0.0):
	return ((time - origo)/period + shift) % 1.

def rebin(x, r):
    m = len(x) // r
    return x[:m*r].reshape((m,r)).mean(axis=1),r

def incl(inputs):
	lgror,lgroa,lgt0,lgper,lgb=inputs[:,-5:].T
	return np.rad2deg(np.arccos(np.exp(lgb)/np.exp(lgroa)))#orbital inclination (in degrees)


class TransitModel(Model):
	parameter_names = ("log_ror", "log_aor", "log_T0", "log_per", "log_b")#, "ecosw", "esinw")

	def get_value(self, t):
		params.t0 = np.exp(self.log_T0) 		#time of inferior conjunction
		params.per = np.exp(self.log_per)	#orbital period
		params.rp = np.exp(self.log_ror)		#planet radius (in units of stellar radii)
		params.a = np.exp(self.log_aor)		#semi-major axis (in units of stellar radii)
		#b=ars*np.cos(np.radians(inc))
		params.inc=np.rad2deg(np.arccos(np.exp(self.log_b)/np.exp(self.log_aor)))#orbital inclination (in degrees)
		params.u = [0.5707,0.1439] 	      	        #limb darkening coefficients #SING 09 for -0.3dex, 4750K logg 3 star (Pretty damn close)
		params.limb_dark = "quadratic"          #limb darkening model

		params.ecc = 0.0				#eccentricity
		params.w = 90				#longitude of periastron (in degrees)

		m=batman.TransitModel(params,t,supersample_factor=5, exp_time=0.5/24.)
		model=m.light_curve(params)
		model=1e6*(model-1)
		return model

muhzconv = 1e6 / (3600*24)
def muhz2idays(muhz):
    return muhz / muhzconv
def muhz2omega(muhz):
    return muhz2idays(muhz) * 2.0 * np.pi
def idays2muhz(idays):
    return idays * muhzconv
def omega2muhz(omega):
    return idays2muhz(omega / (2.0 * np.pi))

###########################################################
loc='./TEST_RESULTS_CELERITE'
if not os.path.isdir(loc):
	os.makedirs(loc)

data=pyfits.open('../kplr009145861_kasoc-psd_llc_v1.fits')
f,p=data[1].data['Frequency'],data[1].data['PSD']

results='../BG_FIT/results.txt'#BG FIT RESULTS
res=np.loadtxt(results,dtype='str')
bgparams=res[:,1].astype(float)
bg=model_func(bgparams,f)




t='../Detrended_full.txt'
#t='../Detrended_phase0.2.txt'
time,lc,elc,_=np.loadtxt(t,unpack=True)#BJD-2454833


idx=np.argsort(time)
time=time[idx]
lc=lc[idx]
elc=elc[idx]

labels=['logS01', 'logomega01', 'logS02', 'logomega02', 'logS0osc', 'logQosc', 'logomega0osc', 'logsigma', 'logror', 'logaor', 'logT0', 'logper', 'logb']
samples=np.load(loc+'/6194_pdfs.npy')
#fig = corner.corner(samples, labels=labels)
#fig.savefig(loc+"/tran_corner.png")
#plt.show()
quantiles = np.percentile(samples,[16,50,84],axis=0).T
logmedians,loguerr,loglerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
lgror,lgroa,lgt0,lgper,lgb=logmedians[-5:]

###########~~RESULTS~~############
for i in range(0,len(labels)):
	print(labels[i],logmedians[i],'+/-',np.mean((loguerr[i],loglerr[i])))
print('\n')
labels2=map(lambda x: x[3:],labels)
labels2[1]='f1'
labels2[3]='f2'
labels2[-7]='fosc'
labels2[7]=r'ln$\sigma$'

samples=np.exp(samples)#out of log
samples[:,7]=np.log(samples[:,7])#leave sigma in log
samples[:,-7]=omega2muhz(samples[:,-7])#convert omega osc to uHz
samples[:,1]=omega2muhz(samples[:,1])#convert omega1 to uHz
samples[:,3]=omega2muhz(samples[:,3])#convert omega2 to uHz


#fig = corner.corner(samples, labels=labels2)
#fig.savefig(loc+"/no_log_tran_corner.png")
#fig.savefig('/home/thomas/Dropbox/PhD/Written Papers/Exoplanet_Host_KOI_6194/no_log_tran_corner.png',dpi=100)
#plt.show()

quantiles = np.percentile(samples,[16,50,84],axis=0).T
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
for i in range(0,len(labels)):
	print(labels2[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))
inc=incl(np.log(samples))
print('inc',np.median(inc),'+/-',np.std(inc))



#########PLOTS##################
final = TransitModel(lgror,lgroa,lgt0,lgper,lgb).get_value(time)
phase=fold(time,np.exp(lgper),np.exp(lgt0),0.5)-0.5
idx=np.argsort(phase)

################################################
###############~FUNKY PLOTS~####################
################################################

#set the GP parameters-FROM SAM RAW
#First granulation
Q = 1.0 / np.sqrt(2.0)
w0 = muhz2omega(13)
S0 = np.var(lc) / (w0*Q)

kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
kernel.freeze_parameter("log_Q") #to make it a Harvey model

#Second granulation
Q = 1.0 / np.sqrt(2.0)
w0 = muhz2omega(35)
S0 = np.var(lc) / (w0*Q)
kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
kernel.freeze_parameter("terms[1]:log_Q") #to make it a Harvey model

#numax
Q = np.exp(3.0)
w0 = muhz2omega(135) #peak of oscillations at 133 uhz
S0 = np.var(lc) / (w0*Q)
kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))

initial = TransitModel(log_ror=np.log(0.02), log_aor=np.log(10),log_T0=np.log(133.9), log_per=np.log(42.3),log_b=np.log(89))
kernel += terms.JitterTerm(log_sigma=0)

gp = celerite.GP(kernel, mean=initial, fit_mean=True)#, log_white_noise=np.log(np.mean(yerr)**2/len(t)), fit_white_noise=False)
gp.compute(time, elc)

gp.set_parameter_vector(logmedians)
#print(gp.get_parameter_vector())

######PSD#############
p2=kernel.get_psd(muhz2omega(f))
p2=p2/(2*np.pi)#ppm^2/Hz (same as end of gatspy)
df=(f[1]-f[0])/1e6

lhs=(1/len(time))*np.sum(lc**2)
rhs= np.sum(p2)
ratio=lhs/rhs#enforce parseval 


p2=p2*ratio/(df*1e6)#ppm^2/uHz

plt.plot(f,p,'k')
plt.plot(f,bg,'b--')
plt.plot(f,p2,'r')
plt.yscale('log')
plt.xscale('log')
plt.show()

sample=np.array([])#final gp+transit -transit
for i in np.arange(len(time) // 8156):
	print(i)
	section = np.arange(i*8156,i*8156 + 8156)
	print('len time:',len(time[section]))
	gp.compute(time[section], elc[section])
	final1 = TransitModel(lgror,lgroa,lgt0,lgper,lgb).get_value(time[section])
	sample=np.append(sample,gp.predict(lc[section],time[section],return_cov=False)-final1)


time=time[:len(sample)]
lc=lc[:len(sample)]
elc=elc[:len(sample)]
final = TransitModel(lgror,lgroa,lgt0,lgper,lgb).get_value(time)
timefine=np.linspace(min(time),max(time),int(1e5))
finalfine = TransitModel(lgror,lgroa,lgt0,lgper,lgb).get_value(timefine)

finalgp=np.array(sample)

phase=fold(time,np.exp(lgper),np.exp(lgt0),0.5)-0.5
idx=np.argsort(phase)


T0,period=np.exp(lgt0),np.exp(lgper)
idxt=[(time>(T0+(7*period))-2) & (time<(T0+(7*period))+2)]
gp.compute(time[idxt],elc[idxt])
mu,var=gp.predict(lc[idxt],time[idxt],return_var=True)#Total model gp+transit
std=np.sqrt(var)

#####################FINAL GP PLOT############################
plt.figure(figsize=(8,10))
plt.subplot(311)
plt.errorbar(time[idxt],lc[idxt],yerr=elc[idxt],fmt='k.',alpha=0.7)#Lightcurve
#plt.plot(time[idxt],finalgp[idxt],alpha=0.5)#Best noise model
plt.plot(time[idxt],final[idxt],'r-',zorder=100,lw=1.5)#Transit model
plt.plot(time[idxt],mu,color='C1',alpha=0.6)#Best noise+transit model
plt.fill_between(time[idxt], mu+std, mu-std, color='C1', alpha=0.4, edgecolor="none")

plt.xlabel('Time (BJD-2454833)')
plt.ylabel('Flux (ppm)')

plt.subplot(312)
plt.plot(phase,lc,'k.',alpha=0.1)
plt.plot(phase[idx],final[idx],'r-')
plt.xlim(-0.2,0.2)
plt.xlabel('Phase')
plt.ylabel('Flux (ppm)')

plt.subplot(313)
plt.plot(phase,lc-finalgp,'k.',alpha=0.1)
plt.plot(phase[idx],final[idx],'r-')
plt.xlim(-0.2,0.2)
plt.xlabel('Phase')
plt.ylabel('Flux (ppm)')
plt.tight_layout()
plt.tight_layout()
plt.savefig(loc+'/Best_gp.pdf')
plt.savefig(loc+'/Best_gp.png',dpi=250)
#plt.savefig('/home/thomas/Dropbox/PhD/Written Papers/Exoplanet_Host_KOI_6194/Best_gp.pdf')
#plt.savefig('/home/thomas/Dropbox/PhD/Written Papers/Exoplanet_Host_KOI_6194/Best_gp.png',dpi=250)
plt.show()

plt.plot(time,lc-finalgp-final,'k.')
plt.show()


print('Mean formal error (ppm)',np.mean(elc))
print('Std residual lightcurve (ppm)',np.std(lc-finalgp-final))
