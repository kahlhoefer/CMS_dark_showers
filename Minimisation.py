import numpy as np
from scipy.special import xlogy
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
import time

verbose = 0 # Set to larger value for additional output

data = np.loadtxt('Data_points_5a.dat') # Manually extracted from HEPData-ins1994864-v3-Figure_5a.csv
bins = data[:,0]/13. #Divide by 13 TeV to get dimensionless argument for fitting function
events = data[:,1]

cross_section_list = np.loadtxt('Theoretical_cross_sections.dat')
cross_sections = interp1d(cross_section_list[:,0],cross_section_list[:,1])

mzplist = [2100, 3100, 4100] #List of all masses, for which histograms are available

signal_dictionary = dict()

for mzp in mzplist:
  signal_dictionary.update({mzp: np.loadtxt('mZp'+str(mzp)+'_mThighRT.txt')[0:65, 1]}) #Load histograms into dictionary
 
guesses = np.loadtxt('GuessList.dat') #List of 71 initial guesses for function minimisation. Designed to ensure good convergence to global minimum
best_p_bg = np.array([-31.66421994, 29.07483373, -24.88109709, -3.68618232])

def fitting_function(x,p0,p1,p2,p3): #Slightly redefined for better numerical stability
  exponent = p0 + p1*x + p2*np.log(x) + p3*np.log(x)**2.
  return np.exp(np.clip(exponent,-100,100))

def log_likelihood(p, signal): #Binned log likelihood for Poisson data
  background = fitting_function(bins, p[0], p[1], p[2], p[3])
  return 2 * np.sum(background + signal - events + xlogy(events, events/(background + signal)) )
  
def best_log_likelihood(signal = np.zeros(65)): #Minimisation routine over background parameters

  res_list = []
  success_list = []

  for p in guesses:
    res = minimize(log_likelihood, p, args=(signal), tol = 0.01)
    if verbose > 2: print (p,res)
    res_list.append(res.fun)
    success_list.append(res.success)
    
  best_ll = min(res_list)
  
  success_rate = np.count_nonzero(success_list) / len(guesses)
  identical_results = np.count_nonzero(np.array(res_list) < best_ll + 0.01) / len(guesses)
  if verbose > 1:
    print('Success rate:', success_rate)
    print('Fraction of results within 0.01 of minimum:', identical_results)
  if success_rate == 0 or identical_results < 0.02: print('Warning! It looks like the minimisation has failed...')
  
  return best_ll

background_log_likelihood = best_log_likelihood()

# For a given signal model, the following code calculates the 95% confidence level upper bound on the signal strength, i.e. it determines the value of mu such that
# -2 * (log L(mu * signal + background) - log L(background)) = 3.84
def signal_strength_bound(signal, fix_params = False):

  if not(fix_params):
  
    def test_statistic(mu):
      result = best_log_likelihood(signal = mu * signal) - background_log_likelihood - 3.84
      if verbose > 0: print(mu, result)
      return result

  else:
  
    def test_statistic(mu):
      result = log_likelihood(best_p_bg, mu * signal) - background_log_likelihood - 3.84
      if verbose > 0: print(mu, result)
      return result
    
  res = root(test_statistic, [1], tol = 0.01)
  return res.x[0]

for mzp in mzplist:
  mubound = signal_strength_bound(signal_dictionary[mzp])
  mubound_fixed = signal_strength_bound(signal_dictionary[mzp], fix_params = True)
  print('Bound on signal strength for M_Zp =',mzp,'GeV: ',mubound, '(', mubound_fixed, 'without background variation)')
  print('Bound on cross section for M_Zp =',mzp,'GeV: ',mubound*cross_sections(mzp),'pb')

