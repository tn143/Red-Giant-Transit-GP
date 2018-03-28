from __future__ import division,print_function
import sys
import os
from os.path import expanduser
import matplotlib.pyplot as plt
import pandas as pd
from ajplanet import pl_rv_array as rv_curve
import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy import optimize
import numpy as np

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def rv_pl(time,params):
	rvsys, K, w, ecc, T, period=params
	w=np.radians(w)
	n=(2*np.pi)/period
	M=n*(time-T)
	E=np.zeros(len(M))
	if ecc==0:
		V=rvsys+K*(np.cos(w+M))
	else:
		for ii,element in enumerate(M): # compute eccentric anomaly
			E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)
		#E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230

		f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V

tran_fit=np.loadtxt('./Tran_fit/TRAN_results.txt',usecols=(1,2,3))
P,Tc,ars,rprs,inc,e,w=tran_fit[:,0].T
print(P)



#Synthetic
labels=['rvsys', 'K', 'w', 'ecc', 'T', 'period']
params=[10,45,w,e,Tc,P]
time=np.random.uniform(0,90,15)
modt=np.linspace(0,90,1e4)

rv=rv_pl(time,params)
modrv=rv_pl(modt,params)
erv=np.random.normal(10,3,len(rv))
rv+=erv*np.random.rand(len(rv))

plt.errorbar(time,rv,yerr=erv,fmt='.')
plt.plot(modt,modrv)
plt.show()

np.savetxt('Fake_RV.txt',np.c_[time,rv,erv])





