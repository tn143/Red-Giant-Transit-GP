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

def rv_pl(params,time):
	rvsys, K, w, ecc, T, period=params
	w=np.radians(w)
	n=(2*np.pi)/period
	M=n*(time-T)
	E=np.zeros(len(M))
	if ecc==0:
		V=rvsys+K*(np.cos(w+M))
	else:
		if len(time)<150:
			E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230
		else:
			for ii,element in enumerate(M): # compute eccentric anomaly
				E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)

		f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))#true anomaly
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V
	
def lnprior(theta):
	if fix_ecc:
		P,Tc,ars,rprs,inc,rvsys,K, Tr,siglc,sigrv = theta		
		b=(ars*np.cos(np.radians(inc)))
	else:
		P,Tc,ars,rprs,inc,rvsys,K, Tr, ecc,w,siglc,sigrv = theta	
		b=(ars*np.cos(np.radians(inc)))*((1-ecc**2)/(1+ecc*np.sin(np.radians(w))))

	loga=np.log10(ars)
	logrp=np.log10(rprs)
	logr=np.log10(rvsys)
	logk=np.log10(K)
	sqrte=np.sqrt(ecc)	

	rho=(3*np.pi*(ars**3))/(G*(P*86400)**2)
	rho_ast=1408*(10.9/134.9)**2#kg/m3
	erho_ast=1.04

	if -100 < rvsys < 100 and 0 < K < 100 and -1<sqrte*np.cos(np.deg2rad(w))<1 and -1<sqrte*np.sin(np.deg2rad(w))<1 and ecc<1 and 17900<Tr<18000\
	and 0<b<1 and 42.1 < P < 42.5 and -1<loga<2 and -3<logrp<0.0 and 133<Tc<135 and 0<siglc<1e3 and 0<sigrv<100:
		lnp_den=-0.5*(rho-rho_ast)**2/(erho_ast**2)
		return lnp_den

	else:
		return -np.inf

def lnlike(theta, time, lc, elc, time_rv, rv, erv):
	if fix_ecc:
		P,Tc,ars,rprs,inc, rvsys, K, Tr = theta
	else:
		P,Tc,ars,rprs,inc,rvsys,K, Tr, ecc,w,siglc,sigrv = theta

	model =get_tran(theta,time)
	inv_sigma2 = 1.0/(elc**2+siglc**2)
	loglike_tran= -0.5*(np.sum((lc-model)**2*inv_sigma2 - np.log(inv_sigma2)))

	modelrv =get_rv(theta,time_rv)
	inv_sigma2 = 1.0/(erv**2+sigrv**2)# + model**2)
	loglike_rv= -0.5*(np.sum((rv-modelrv)**2*inv_sigma2 - np.log(inv_sigma2)))

	return loglike_tran+loglike_rv


def lnprob(theta, time, lc, elc, time_rv, rv, erv):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, time, lc, elc, time_rv, rv, erv)

def get_tran(para,time):
	if fix_ecc:
		P,Tc,ars,rprs,inc, rvsys, K, Tr = para
	else:
		P,Tc,ars,rprs,inc,rvsys,K, Tr, e,w,siglc,sigrv = para

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

def get_rv(para,time):
	if fix_ecc:
		P,Tc,ars,rprs,inc, rvsys, K, Tr = para
		e=0.0
		w=90
	else:
		P,Tc,ars,rprs,inc,rvsys,K, Tr, e,w,siglc,sigrv = para

	prv=[rvsys, K, w, e, Tr, P]
	model=rv_pl(prv,time)
	return model


if __name__ == "__main__":
	loc=home+'/Dropbox/PhD/Year_4/Year_3_koi6194_kic9145861/tran_rv_fit'
	if not os.path.isdir(loc):
		os.makedirs(loc)

	#t='./Detrended_full.txt'
	t='./Detrended_0.2.txt'
	time,lc,elc=np.loadtxt(t,unpack=True)
	idx=np.argsort(time)
	time=time[idx]
	lc=lc[idx]
	elc=elc[idx]

	time_rv,rv,erv=np.loadtxt('./koi6194.txt',unpack=True,usecols=(0,1,2),skiprows=1)
	params = batman.TransitParams()
	fix_ecc=False

	labels=['P','Tc','a/R','Rp/R','inc','rvsys','K','T0','e','w','sig-lc','sig-rv']
	initial=[42.29,133.9,10,0.013,89,9,43,17925,0.1,90,10,10]
	if fix_ecc:
		labels=labels[:-2]
		initial=initial[:-2]


	plt.subplot(211)
	plt.plot(time,lc,'k.',alpha=0.4)
	plt.plot(time,get_tran(initial,time))
	plt.subplot(212)
	plt.errorbar(time_rv,rv,erv,fmt='.')
	time_mod=np.arange(time_rv.min(),time_rv.max(),0.5)
	plt.plot(time_mod,get_rv(initial,time_mod))
	plt.savefig(loc+'/initial.png')
	plt.show()



	nwalkers, niter, ndim = 20, 500, len(labels)
	burnin=int(niter/2)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, lc, elc,time_rv,rv,erv))

	p0 = np.zeros([nwalkers, ndim])
	for j in range(nwalkers):
		p0[j,:] = initial + 1e-1*np.random.randn(ndim)

	print('... burning in ...')
	for p, lnprob, rvstate in tqdm(sampler.sample(p0, iterations=niter),total=niter):
		pass
		# Clear and run the production chain.
		#sampler.reset()
		#np.save(loc+'/3890_pdfs.npy',samples)
	print('... running sampler ...')
	for p, lnprob, rvstate in tqdm(sampler.sample(p, lnprob0=lnprob,iterations=niter),total=niter):
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

	fig = corner.corner(samples, labels=labels)
	fig.savefig(loc+"/t+rv_corner.png")
	plt.show()

	quantiles = np.percentile(samples,[16,50,84],axis=0).T 
	medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
	np.savetxt(loc+'/TRAN+RV_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')
	for i in range(0,len(labels)):
		print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

	if fix_ecc:
		P,Tc,ars,rprs,inc, rvsys, K, Tr,siglc,sigrv = medians
	else:
		P,Tc,ars,rprs,inc,rvsys,K, Tr, e,w,siglc,sigrv = medians

	def fold(time, period, origo=0.0, shift=0.0):
		return ((time - origo)/period + shift) % 1.

	#Tran
	start=get_tran(initial,time)
	final=get_tran(medians,time)

	#Rv
	time_mod=np.arange(time_rv.min(),time_rv.max(),0.5)

	startrv = get_rv(initial,time_mod)
	finalrv = get_rv(medians,time_mod)

	plt.subplot(311)
	plt.plot(time,lc,'k.',alpha=0.4)
	plt.plot(time,start)
	plt.plot(time,final)

	phase=fold(time,P,Tc,0.5)-0.5
	idx=np.argsort(phase)

	plt.subplot(312)
	plt.plot(phase,lc,'k.',alpha=0.5)
	plt.plot(phase[idx],final[idx],'r--',lw=2)
	#########

	phase_rv=((time_rv-Tr)/P)%1
	phase_mod=((time_mod-Tr)/P)%1

	plt.subplot(313)

	idx=np.argsort(phase_mod)
	plt.errorbar(phase_rv,rv,erv,fmt='k.')
	plt.plot(phase_mod[idx],startrv[idx],'b')
	plt.plot(phase_mod[idx],finalrv[idx],'r')
	plt.savefig(loc+'/fin.png')
	plt.show()






