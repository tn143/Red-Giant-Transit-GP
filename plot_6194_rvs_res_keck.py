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
def rv_pl(time,params):
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

		f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V


def fold(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5


time,rv,erv=np.loadtxt('./koi6194.txt',unpack=True,usecols=(0,1,2),skiprows=1)#keck JD - 2430000
#time=time+2440000
time2,rv2,erv2=np.loadtxt('./K06194.ccfSum.txt',unpack=True,usecols=(0,1,2),skiprows=1)#tres-BJD_UTC 

medians=np.loadtxt('./RV_results.txt',usecols=(1,),skiprows=1)
rvsys, K, w, ecc, T0, period,sig=medians

time3=np.append(time,time2)
rv3=np.append(rv,rv2)
erv3=np.append(erv,erv2)

p=42.2949943346
t0=13.6686429628

phase=fold(time,period,T0,0.0)
plt.errorbar(phase,rv,yerr=np.sqrt(sig**2+erv**2),fmt='.')
plt.axhline(y=0,ls='--',c='k')
plt.show()


mod_time=np.linspace(min(time),max(time),1e4)
model=rv_pl(mod_time,medians[:-1])

pha=fold(time,period,T0,0.5)
phase_mod=fold(mod_time,period,T0,0.5)
idx=np.argsort(phase_mod)

plt.errorbar(pha,rv,yerr=np.sqrt(sig**2+erv**2),fmt='.')
plt.plot(phase_mod[idx],model[idx],'r')
plt.xlabel('Phase',fontsize=20)
plt.ylabel(r'Radial Velocity (ms$^-1$)',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(-100,100)
plt.xlim(-0.5,0.5)
plt.tight_layout()
plt.savefig('6194_rv.pdf')
plt.savefig('6194_rv.png',dpi=250)
plt.show()





