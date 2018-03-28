#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner
from os.path import expanduser
import sys
from scipy.optimize import fsolve
import pandas as pd
import gatspy.periodic as gp
from tqdm import tqdm
from time import sleep
from scipy import stats
import os

home=expanduser('~')

def phase(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))
def tru_anom(params,time):
	rvsys, K, w, ecc, Tr, P,sig2 = params
	w=np.radians(w)
	n=(2*np.pi)/P
	M=n*(time-Tr)#if e==0 then E==M
	E=np.zeros(len(M))

	if len(time)<150:
		E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230
	else:
		for ii,element in enumerate(M): # compute eccentric anomaly
			E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)

	#f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))
	f=2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(0.5*E))

	return f	

def rv_pl(time,params):
	rvsys, K, w, ecc, T, period,sig2=params
	w=np.radians(w)
	#if w<0:
	#	w=w+(2*np.pi)
	if ecc<0:
		ecc=np.abs(ecc)

	if ecc==0:
		w=np.radians(w)
		n=(2*np.pi)/P
		M=n*(time-Tr)
		E=np.zeros(len(M))
		V=rvsys+K*(np.cos(M))
	else:
		f=tru_anom(params,time)
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V

# Define the probability function as likelihood * prior.
def lnprior(theta):
	rvsys, K, w, ecc, T0, period,sig2 = theta
	logp=np.log10(period)
	sqrte=np.sqrt(ecc)

	#logr=np.log10(rvsys)
	logk=np.log10(K)
	if -100 < rvsys < 100 and -2 < logk < 2 and -1<sqrte*np.cos(np.deg2rad(w))<1 and -1<sqrte*np.sin(np.deg2rad(w))<1 and ecc<1 and 0<w<360\
	 and 58000<T0<58045 and 42<period<42.5 and 0<sig2<10:
		lnp_per=-0.5*(period-42.295)**2/(0.002**2)#42.294739±0.001569
		return lnp_per

	else:
		return -np.inf

def lnlike(theta, time, rv, erv):
    rvsys, K, w, ecc, T0, period,sig2 = theta
    model =rv_pl(time,theta)

    inv_sigma2 = 1.0/(sig2**2+erv**2)
    return -0.5*(np.sum((rv-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, time, rv, erv):
    lp = lnprior(theta)
    if not np.isfinite(lp):
		return -np.inf
    return lp + lnlike(theta, time, rv, erv)


loc='./RV_fit_free_ecc_sig2'
if not os.path.isdir(loc):
	os.makedirs(loc)

time,rv,erv=np.loadtxt('./koi6194.txt',unpack=True,usecols=(0,1,2),skiprows=1)#JD-2440000
time+=40000#JD-2400000

#rvsys, K, w, ecc, T0, period
labels=['rvsys', 'K', 'w', 'ecc', 'T0', 'period','sig2']
initial=[0,40,90,0.01,58020,42.3,1]

mod_time=np.arange(min(time),max(time),1)
plt.errorbar(time,rv,erv,fmt='.')
plt.plot(mod_time,rv_pl(mod_time,initial))
plt.savefig(loc+'/initial.png')
plt.show()

# Set up the sampler.
nwalkers, niter, ndim = 100, 3000, len(labels)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,rv,erv))

p0 = np.zeros([nwalkers, ndim])

for j in range(nwalkers):
	p0[j,:] = initial + 1e-2*np.random.randn(ndim)

print('... burning in ...')
#for p, lnprob, state in sampler.sample(p0, iterations=niter):
for p, lnprob, state in tqdm(sampler.sample(p0, iterations=niter),total=niter):
	sleep(0.001)

#Find best walker and resamples around it
p = p[np.argmax(lnprob)]
p0 = np.zeros([nwalkers, ndim])
for j in range(nwalkers):
	p0[j,:] = p + 1e-3*np.random.randn(ndim)

# Clear and run the production chain.
#sampler.reset()
print('... running sampler ...')
#for p, lnprob, state in sampler.sample(p, lnprob0=lnprob,iterations=niter):
for p, lnprob, state in tqdm(sampler.sample(p0, lnprob0=lnprob,iterations=niter),total=niter):
	sleep(0.001)

fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8, 9))
burnin=int(niter/2)

for i in range(0,len(initial)):
	axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
plt.savefig(loc+'/chain.png')
plt.show()
#print('n temps:', ntemps, "log evidence: ", sampler.thermodynamic_integration_log_evidence())

# Make the corner plot.
burnin = int(niter/2)
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=labels)
plt.savefig(loc+'/corner.png')
plt.show()

np.save(loc+'/samples.npy',samples)

plt.errorbar(time,rv,erv,fmt='.')
for theta in samples[np.random.randint(len(samples), size=int(niter/25))]:
	plt.plot(mod_time,rv_pl(mod_time,theta),c='k',alpha=0.1)
plt.savefig(loc+'/rv_solutions.png')
plt.show()


quantiles = np.percentile(samples,[16,50,84],axis=0).T 
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-

np.savetxt(loc+'/RV_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')

for i in range(0,len(labels)):
	print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

rvsys, K, w, ecc, T0, period,sig2=medians
plt.errorbar(time,rv,erv,fmt='.')
plt.plot(mod_time,rv_pl(mod_time,medians))
plt.savefig(loc+'/final.png')
plt.show()


