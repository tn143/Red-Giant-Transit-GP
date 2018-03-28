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

home=expanduser('~')

def phase(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def rv_pl(time,params):
	rvsys, K, w, ecc, T, period=params
	w=np.radians(w)
	if w<0:
		w=w+(2*np.pi)
	if ecc<0:
		ecc=np.abs(ecc)

	n=(2*np.pi)/period
	M=n*(time-T)
	E=np.zeros(len(M))
	for ii,element in enumerate(M): # compute eccentric anomaly
		E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)
	f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))

	V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V

# Define the probability function as likelihood * prior.
def lnprior(theta):
	rvsys, K, w, ecc, T0, period = theta
	logp=np.log10(period)
	logr=np.log10(rvsys)
	logk=np.log10(K)
	if -1 < logr < 4 and -2 < logk < 3 and 0.0<w<360 and 0<T0<1e6 and 0.1<logp<2:
		ln_e=stats.beta(0.867,3.03)
		return ln_e.logpdf(ecc)
	else:
		return -np.inf

def lnlike(theta, time, rv, erv):
    rvsys, K, w, ecc, T0, period = theta
    model =rv_pl(time,theta)

    inv_sigma2 = 1.0/(erv**2)# + model**2)
    return -0.5*(np.sum((rv-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, time, rv, erv):
    lp = lnprior(theta)
    if not np.isfinite(lp):
		return -np.inf
    return lp + lnlike(theta, time, rv, erv)

filerv='../Fake_RV.txt'

time,rv,erv=np.loadtxt(filerv,unpack=True)
#time=(time-time[0])
#time=time-2400000
#rv=rv/1000
#erv=erv/1000

mod_time=np.linspace(min(time),max(time),1e4)

from gatspy.periodic import LombScargleFast
from gatspy.periodic import LombScargle
N = 10000
periods=np.linspace(10,2*max(time-time[0]),N)
fmin = 1. / periods.max()
fmax = 1. / periods.min()
df = (fmax - fmin) / N

model = LombScargle().fit(time-time[0], rv, erv)
power = model.score_frequency_grid(fmin, df, N)
freqs = fmin + df * np.arange(N)

periods=1./freqs
# plot the results
plt.plot(1. / freqs, power)
plt.show()

#rvsys, K, w, ecc, T0, period
labels=['rvsys', 'K', 'w', 'ecc', 'T0', 'period']
initial=[np.mean(rv),np.std(rv-np.mean(rv)),90,0.2,time[rv==min(rv)],periods[power==max(power)]]

print('guess p: ', periods[power==max(power)])
print('guess sys v: ',np.mean(rv))
print('guess K: ',np.std(rv-np.mean(rv)))
print(np.c_[labels,initial])


plt.errorbar(time,rv,erv,fmt='.')
plt.plot(mod_time,rv_pl(mod_time,initial))
plt.show()

# Set up the sampler.
ntemps, nwalkers, niter, ndim = 4, 250, 750, len(labels)
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior, loglargs=(time, rv, erv))
p0 = np.zeros([ntemps, nwalkers, ndim])
for i in range(ntemps):
    for j in range(nwalkers):
	p0[i,j,:] = initial + 1e-2*np.random.randn(ndim)

print('... burning in ...')
for p, lnprob, lnlike in tqdm(sampler.sample(p0, iterations=niter),total=niter):
	sleep(0.001)

#print(np.shape(p0))
#print(np.shape(p))
# Clear and run the production chain.
sampler.reset()

print('... running sampler ...')
for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike,iterations=niter),total=niter):
	sleep(0.001)

fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8, 9))

for i in range(0,len(initial)):
	axes[i].plot(sampler.chain[0, :, :, i].T, color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
plt.show()
#print('n temps:', ntemps, "log evidence: ", sampler.thermodynamic_integration_log_evidence())

# Make the corner plot.
burnin = (niter/2)
samples = sampler.chain[0, :, burnin:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=labels)
plt.show()

quantiles = np.percentile(samples,[16,50,84],axis=0).T 
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-

np.savetxt('RV_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')

for i in range(0,len(labels)):
	print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

rvsys, K, w, ecc, T0, period=medians
plt.errorbar(time,rv,erv,fmt='.')
plt.plot(mod_time,rv_pl(mod_time,medians))
plt.show()


