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


G=constants.G
home=expanduser('~')
#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))
params = batman.TransitParams()
def fold(time, period, origo=0.0, shift=0.0):
	return ((time - origo)/period + shift) % 1.

def rebin(x, r):
    m = len(x) // r
    return x[:m*r].reshape((m,r)).mean(axis=1),r


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

		m=batman.TransitModel(params,time,supersample_factor=5, exp_time=0.5/24.)
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

#print(np.log(muhz2omega(3)), np.log(muhz2omega(50)))

###########################################################
loc='./TEST_RESULTS_CELERITE'

if not os.path.isdir(loc):
	os.makedirs(loc)

t='../Detrended_full.txt'
#t='../Detrended_phase0.2.txt'
time,lc,elc,_=np.loadtxt(t,unpack=True)#BJD-2454833

idx=np.argsort(time)
time=time[idx]
lc=lc[idx]
elc=elc[idx]

#plt.plot(time,lc,'k.',alpha=0.2)
#plt.show()

#set the GP parameters-FROM SAM RAW
#First granulation
Q = 1.0 / np.sqrt(2.0)
w0 = muhz2omega(13)
S0 = np.var(lc) / (w0*Q)

kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                       bounds=[(0, 20), (-15, 15), (0.3, 4)]) #omega upper bound: 275 muhz
kernel.freeze_parameter("log_Q") #to make it a Harvey model

#Second granulation
Q = 1.0 / np.sqrt(2.0)
w0 = muhz2omega(35)
S0 = np.var(lc) / (w0*Q)
kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                   bounds=[(0, 20), (-15, 15), (np.log(muhz2omega(30)), np.log(muhz2omega(1000)))])
kernel.freeze_parameter("terms[1]:log_Q") #to make it a Harvey model

#numax
Q = np.exp(3.0)
w0 = muhz2omega(135) #peak of oscillations at 133 uhz
S0 = np.var(lc) / (w0*Q)
kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                      bounds=[(0, 20), (0.5, 4.2), (np.log(muhz2omega(100)), np.log(muhz2omega(200)))])

#initial guess of transit model
#parameter_names = ("log_ror", "log_aor", "log_T0", "log_per", "log_inc")#, "ecosw", "esinw")

initial = TransitModel(log_ror=np.log(0.02), log_aor=np.log(10),
                    log_T0=np.log(133.9), log_per=np.log(42.3),
                    log_b=np.log(0.8),
                    bounds=[(-5, 0), (0,3.0), (np.log(133), np.log(134.5)), (np.log(42.2),np.log(42.4)),(-30,0)])

#initial.freeze_parameter("log_per")

kernel += terms.JitterTerm(log_sigma=0, bounds=[(-20,20)])

gp = celerite.GP(kernel, mean=initial, fit_mean=True)
gp.compute(time, elc)


#find max likelihood params
from scipy.optimize import minimize

def neg_log_like(params, lc, gp):
    gp.set_parameter_vector(params)
    ll = gp.log_likelihood(lc)
    if not np.isfinite(ll):
        return 1e10
    return -ll

initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()

print('Fitting max L')
r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(lc, gp))
gp.set_parameter_vector(r.x)

#FIRST FIT-Max L

lgror,lgroa,lgt0,lgper,lgb=gp.get_parameter_vector(include_frozen=True)[-5:]
print('Max L results',np.exp([lgror,lgroa,lgt0,lgper,lgb]))

final = TransitModel(lgror,lgroa,lgt0,lgper,lgb).get_value(time)
phase=fold(time,np.exp(lgper),np.exp(lgt0),0.5)-0.5
idx=np.argsort(phase)

ph=phase[idx]
phf=lc[idx]

ph=rebin(ph,50)[0]
phf=rebin(phf,50)[0]

plt.plot(phase,lc,'k.',alpha=0.05)
plt.plot(ph,phf,'b.')
plt.plot(phase[idx],final[idx],'r--')
plt.show()

plt.plot(time,lc,'.')
plt.plot(time,final)
plt.show()

print(gp.get_parameter_bounds(include_frozen=True))
########################################
########################################
########################################

#####################EMCEE Fitting for parameter uncertainties##################
def lnprob(params, y, gp):
	gp.set_parameter_vector(params)
	lgror,lgroa,lgt0,lgper,lgb=gp.get_parameter_vector(include_frozen=True)[-5:]
	ars=np.exp(lgroa)
	P=np.exp(lgper)

	#Add b prior
	#b=(ars*np.cos(np.radians(inc)))

	#Add density prior
	rho=(3*np.pi*(ars**3))/(G*(P*86400)**2)
	rho_ast=1408*(10.9/134.9)**2#kg/m3
	erho_ast=1.04
	lnp_den=-0.5*(rho-rho_ast)**2/(erho_ast**2)
	lp = gp.log_prior()
	if not np.isfinite(lp):
	    return -np.inf
	ll = gp.log_likelihood(y)

	if not np.isfinite(ll):
	    return -np.inf
	return ll + lp + lnp_den


initial = gp.get_parameter_vector()
labels=gp.get_parameter_names()

#labels=map(lambda x: x.split(':')[-1],labels)
labels=['logS01', 'logomega01', 'logS02', 'logomega02', 'logS0osc', 'logQosc', 'logomega0osc', 'logsigma', 'logror', 'logaor', 'logT0', 'logper', 'logb']

nwalkers, niter, ndim = 100, 2500, len(labels)
burnin=int(niter/2)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lc,gp))


p0 = np.zeros([nwalkers, ndim])
for j in range(nwalkers):
	p0[j,:] = initial + 5e-3*np.random.randn(ndim)

print('... burning in ...')
for p, lnprob, rvstate in tqdm(sampler.sample(p0, iterations=niter),total=niter):
	pass
#Find best walker and resamples around it
p = p[np.argmax(lnprob)]
print(np.exp(p))#values at end of burn
p0 = np.zeros([nwalkers, ndim])

for j in range(nwalkers):
	p0[j,:] = p + 1e-4*np.random.randn(ndim)
sampler.reset()
print('... running sampler ...')
for p, lnprob, rvstate in tqdm(sampler.sample(p0,iterations=niter),total=niter):
	pass

##################EMCEE DONE#########################
################PRETTY PLOTS NOW###############

fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8, 9))

for i in range(0,len(initial)):
	axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
fig.savefig(loc+"/t+rv_chains.png")
plt.show()
#print('n temps:', ntemps, "log evidence: ", sampler.thermodynamic_integration_log_evidence())

# Make the corner plot.
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
np.save(loc+'/6194_pdfs.npy',samples)#Save out samples for other code

fig = corner.corner(samples, labels=labels)
fig.savefig(loc+"/tran_corner.png")#still in log space parameters
plt.show()

quantiles = np.percentile(samples,[16,50,84],axis=0).T
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
lgror,lgroa,lgt0,lgper,lgb=medians[-5:]


#np.savetxt(loc+'/TRAN_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')
for i in range(0,len(labels)):
	print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

print('\n')
labels2=map(lambda x: x[3:],labels)
quantiles = np.percentile(np.exp(samples),[16,50,84],axis=0).T
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
for i in range(0,len(labels)):
	print(labels2[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))


final = TransitModel(lgror,lgroa,lgt0,lgper,lgb).get_value(time)
phase=fold(time,np.exp(lgper),np.exp(lgt0),0.5)-0.5
idx=np.argsort(phase)

ph=phase[idx]
phf=lc[idx]

ph=rebin(ph,50)[0]
phf=rebin(phf,50)[0]


plt.plot(phase,lc,'k.',alpha=0.05)
plt.plot(ph,phf,'b.')
plt.plot(phase[idx],final[idx],'r--')
plt.xlim(-0.3,0.3)
plt.show()

plt.plot(time,lc,'.')
plt.plot(time,final)
plt.show()

