from __future__ import division,print_function
import numpy as np
import sys
import os
from os.path import expanduser
import pyfits
import matplotlib.pyplot as plt

home=expanduser('~')
data=pyfits.open(home+'/Dropbox/PhD/Year_4/Year_3_koi6194_kic9145861/kplr009145861_kasoc-psd_llc_v1.fits')
f,p=data[1].data['Frequency'],data[1].data['PSD']

plt.plot(f,p,'k')
plt.xlim(1,280)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency ($\mu$Hz)', fontsize=14)
plt.ylabel('Power Spectral Density (ppm$^2\mu$Hz$^{-1}$)', fontsize=14)
plt.tight_layout()
plt.savefig('koi6194_psd_kasoc.pdf')
plt.show()

