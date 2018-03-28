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

def fold(time, period, origo=0.0, shift=0.5):
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
	if w<0:
		w=w+(2*np.pi)
	if ecc<0:
		ecc=np.abs(ecc)

	if ecc==0:
		w=np.radians(w)
		n=(2*np.pi)/period
		M=n*(time-T)
		E=np.zeros(len(M))
		V=rvsys+K*(np.cos(M))
	else:
		f=tru_anom(params,time)
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V



loc='./RV_fit_free_ecc_sig2'
if not os.path.isdir(loc):
	os.makedirs(loc)

time,rv,erv=np.loadtxt('./koi6194.txt',unpack=True,usecols=(0,1,2),skiprows=1)#JD-2440000
time+=40000#JD-2400000
#time-=54833#BKJD
#rvsys, K, w, ecc, T0, period
labels=['rvsys', 'K', 'w', 'ecc', 'T0', 'period','sig2']

#medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
medians=np.loadtxt(loc+'/RV_results.txt',usecols=(1,))


loc2='./RV_fit_fix_ecc_sig2'
noecc=np.loadtxt(loc2+'/RV_results.txt',usecols=(1,))
noecc=noecc[:2].tolist()+[90,0]+noecc[-3:].tolist()
for i in range(0,len(labels)):
	print(labels[i],medians[i],noecc[i])


mod_time=np.arange(min(time)-10,max(time)+10,0.1)
final=rv_pl(mod_time,medians)

plt.figure(figsize=(16,6))
plt.errorbar(time,rv,np.sqrt(erv**2+12**2),fmt='k.')#sigma term added in qyad
plt.plot(mod_time,final,'r-',label=r'$e=0.4$')

final_ecc0=rv_pl(mod_time,noecc)
plt.plot(mod_time,final_ecc0,'b:',label=r'$e=0$')

#for i in range(58000,58100,1):
#	noecc[4]=i
#	print(i,noecc[4])

plt.legend(loc='best',fontsize=14)
plt.xlim(min(mod_time),max(mod_time))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Time',fontsize=18)
plt.ylabel('Radial Velocity (ms$^{-1}$',fontsize=18)
plt.tight_layout()
plt.savefig('./Celerite/CIVIL_SERVICE/RV.png',dpi=200)
plt.show()
sys.exit()

rvsys, K, w, ecc, T0, period=medians
phaserv=fold(time,period,T0)
phase=fold(mod_time,period,T0)

idx=np.argsort(phase)

plt.errorbar(phaserv,rv,erv,fmt='k.')
plt.plot(phase_mod[idx],final[idx],'r--')
plt.xlim(-0.5,0.5)
plt.xlabel('Phase')
plt.ylabel('Radial Velocity (ms$^{-1}$')

plt.tight_layout()
plt.savefig(loc+'/rv_solutions.pdf')
plt.show()


