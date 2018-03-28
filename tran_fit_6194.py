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
from scipy import stats
import batman
import pandas as pd
import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy import optimize
from tqdm import tqdm
import os
from scipy import constants


G=constants.G
home=expanduser('~')
#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def lnprior(theta):
	if fix_ecc:
		P,Tc,ars,rprs,inc,sig2= theta		
		b=(ars*np.cos(np.radians(inc)))
	else:
		P,Tc,ars,rprs,inc, ecc,w,sig2 = theta	
		b=(ars*np.cos(np.radians(inc)))*((1-ecc**2)/(1+ecc*np.sin(np.radians(w))))
		sqrte=np.sqrt(ecc)	
	loga=np.log10(ars)
	logrp=np.log10(rprs)
	

	rho=(3*np.pi*(ars**3))/(G*(P*86400)**2)
	rho_ast=1408*(10.9/134.9)**2#kg/m3
	erho_ast=1.04

	#42.2958069574 T0 133.925815091


	if 0<b<1 and 42.2 < P < 42.4 and 0<loga<2 and -3<logrp<-1 and 133.5<Tc<134.5 and 0<sig2<500:
		lnp_den=-0.5*(rho-rho_ast)**2/(erho_ast**2)
		if fix_ecc:
			return lnp_den
		else:
			if -1<sqrte*np.cos(np.deg2rad(w))<1 and -1<sqrte*np.sin(np.deg2rad(w))<1 and ecc<1:
				return lnp_den
			else:
				return -np.inf

	else:
		return -np.inf

def lnlike(theta, time, lc, elc):
	if fix_ecc:
		P,Tc,ars,rprs,inc,sig2 = theta
	else:
		P,Tc,ars,rprs,inc,ecc,w,sig2= theta

	model =get_tran(theta,time)
	inv_sigma2 = 1.0/(elc**2 +sig2**2)
	loglike_tran= -0.5*(np.sum((lc-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	return loglike_tran


def lnprob(theta, time, lc, elc):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, time, lc, elc)

def get_tran(para,time):
	if fix_ecc:
		P,Tc,ars,rprs,inc,sig2= para
	else:
		P,Tc,ars,rprs,inc, e,w,sig2= para

	params.t0 = Tc 				#time of inferior conjunction 
	params.per = P				#orbital period	
	params.rp = rprs				#planet radius (in units of stellar radii)
	params.a = ars				#semi-major axis (in units of stellar radii)
	params.inc = inc			#orbital inclination (in degrees)	
	params.u = [0.5707,0.1439] 	      	        #limb darkening coefficients #SING 09 for -0.3dex, 4750K logg 3 star (Pretty damn close)
	params.limb_dark = "quadratic"          #limb darkening model
	if fix_ecc:
		params.ecc = 0.0				#eccentricity	
		params.w = 90				#longitude of periastron (in degrees) 
	else:
		params.ecc=e
		params.w=w

	m=batman.TransitModel(params,time,supersample_factor=5, exp_time=0.5/24.)
	model=m.light_curve(params)
	model=1e6*(model-1)
	return model


if __name__ == "__main__":
	loc=home+'/Dropbox/PhD/Year_4/Year_3_koi6194_kic9145861/Tran_fit'
	#loc=home+'/Dropbox/PhD/Year_4/Year_3_koi6194_kic9145861/Tran_fit_free_ecc'

	if not os.path.isdir(loc):
		os.makedirs(loc)

	#t='./Detrended_full.txt'
	t='./Detrended_phase0.2.txt'
	time,lc,elc,_=np.loadtxt(t,unpack=True)#BJD-2454833

	print(np.std(lc),np.mean(elc))
	print(np.sqrt(np.std(lc)**2-np.mean(elc)**2))

	print(min(time))
	idx=np.argsort(time)
	time=time[idx]
	lc=lc[idx]
	elc=elc[idx]

	params = batman.TransitParams()
	fix_ecc=True

	labels=['$P$',r'$T_{0}$',r'$a/R_{\star}$',r'$R_{P}/R_{\star}$',r'$i^{\circ}$','$e$','$w$',r'$\sigma_{lc}$']#42.2958069574 T0 133.925815091
	initial=[42.25,133.6,8,0.009,88,0.1,90,100]
	if fix_ecc:
		labels=labels[:5]+labels[-1:]
		initial=initial[:5]+initial[-1:]

	for i in range(0,len(labels)):
		print(labels[i],initial[i])



	plt.plot(time,lc,'k.',alpha=0.4)
	plt.plot(time,get_tran(initial,time))
	plt.savefig(loc+'/initial.png')
	plt.show()


	nwalkers, niter, ndim = 100, 2000, len(labels)
	burnin=int(niter/2)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, lc, elc))

	p0 = np.zeros([nwalkers, ndim])
	for j in range(nwalkers):
		p0[j,:] = initial + 5e-2*np.random.randn(ndim)

	print('... burning in ...')
	for p, lnprob, rvstate in tqdm(sampler.sample(p0, iterations=niter),total=niter):
		pass
		# Clear and run the production chain.
		#sampler.reset()
		#np.save(loc+'/3890_pdfs.npy',samples)

	#Find best walker and resamples around it
	p = p[np.argmax(lnprob)]
	print(p)
	p0 = np.zeros([nwalkers, ndim])

	for j in range(nwalkers):
		p0[j,:] = p + 1e-3*np.random.randn(ndim)


	sampler.reset()

	print('... running sampler ...')
	for p, lnprob, rvstate in tqdm(sampler.sample(p0,iterations=niter),total=niter):
		pass

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
	np.save(loc+'/6194_pdfs.npy',samples)

	fig = corner.corner(samples, labels=labels)
	fig.savefig(loc+"/tran_corner.png")
	plt.show()
	
	quantiles = np.percentile(samples,[16,50,84],axis=0).T 
	medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
	np.savetxt(loc+'/TRAN_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')
	for i in range(0,len(labels)):
		print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

	if fix_ecc:
		P,Tc,ars,rprs,inc,sig2 = medians
	else:
		P,Tc,ars,rprs,inc, e,w,sig2 = medians

	def fold(time, period, origo=0.0, shift=0.0):
		return ((time - origo)/period + shift) % 1.

	#Tran
	start=get_tran(initial,time)
	final=get_tran(medians,time)

	plt.subplot(211)
	plt.plot(time,lc,'k.',alpha=0.4)
	plt.plot(time,start)
	plt.plot(time,final)

	phase=fold(time,P,Tc,0.5)-0.5
	idx=np.argsort(phase)

	plt.subplot(212)
	plt.plot(phase,lc,'k.',alpha=0.5)
	plt.plot(phase[idx],final[idx],'r--',lw=2)
	plt.savefig(loc+'/fin.png')
	plt.show()






