import matplotlib.pyplot as plt
import numpy as np

theo = np.loadtxt('Theoretical_cross_sections.dat')
observed = np.loadtxt('Observed_cross_sections.dat')
analysis = np.loadtxt('cross_section_bound.dat')

plt.plot(theo[:,0],theo[:,1],label='Theoretical cross section')
plt.plot(observed[:,0]*1000,observed[:,1],'.',label='Observed bound (CMS)')
plt.plot(analysis[:,0],analysis[:,1],label='Observed bound (this work)')

plt.yscale('log')
plt.ylim((1e-3,10))
plt.xlim((1400,5200))
plt.xlabel(r"$m_{Z'}$ [GeV]")
plt.ylabel(r"$\sigma \times \mathrm{BR}$ [pb]")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Fig.pdf')
plt.show()
