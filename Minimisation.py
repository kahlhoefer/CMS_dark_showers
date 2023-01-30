import numpy as np
from scipy.special import xlogy
from scipy.optimize import minimize, root, brentq
from scipy.interpolate import interp1d
from scipy.stats import norm
import time

verbose = 0 # Set to larger value for additional output

data5a = np.loadtxt('Data_points_5a.dat') # Manually extracted from HEPData-ins1994864-v3-Figure_5a.csv
data5b = np.loadtxt('Data_points_5b.dat') # Manually extracted from HEPData-ins1994864-v3-Figure_5a.csv
bins5a = data5a[:,0]/13. #Divide by 13 TeV to get dimensionless argument for fitting function
events5a = data5a[:,1]
bins5b = data5b[:,0]/13. #Divide by 13 TeV to get dimensionless argument for fitting function
events5b = data5b[:,1]

cross_section_list = np.loadtxt('Theoretical_cross_sections.dat')
cross_sections = interp1d(cross_section_list[:,0],cross_section_list[:,1])

mzplist = [2100, 3100, 4100] #List of all masses, for which histograms are available

signal_dictionary_high = dict()
signal_dictionary_low = dict()

for mzp in mzplist:
  signal_dictionary_high.update({mzp: np.loadtxt('mZp'+str(mzp)+'_100k_mThighRT.txt')[0:65, 1]}) #Load histograms into dictionary
  signal_dictionary_low.update({mzp: np.loadtxt('mZp'+str(mzp)+'_100k_mTlowRT.txt')[0:65, 1]}) #Load histograms into dictionary
  
guesses5a = np.loadtxt('GuessList_5a.dat') #List of 71 initial guesses for function minimisation. Designed to ensure good convergence to global minimum
guesses5b = np.loadtxt('GuessList_5b.dat') #List of 70 initial guesses for function minimisation. Designed to ensure good convergence to global minimum

best_p_bg_5a = np.array([-31.66421994, 29.07483373, -24.88109709, -3.68618232])
best_p_bg_5b = np.array([-0.999156, -9.86746, -5.77419])

def fitting_function_5a(x,p0,p1,p2,p3): #Slightly redefined for better numerical stability
  exponent = p0 + p1*x + p2*np.log(x) + p3*np.log(x)**2.
  return np.exp(np.clip(exponent,-100,100))

def fitting_function_5b(x,p0,p1,p2): #Slightly redefined for better numerical stability
  exponent = p0 + p1*x + p2*np.log(x)
  return np.exp(np.clip(exponent,-100,100))

def log_likelihood_5a(p, signal, Asimov): #Binned log likelihood for Poisson data
  background = fitting_function_5a(bins5a, p[0], p[1], p[2], p[3])
  if not(Asimov): events = events5a
  else: events = fitting_function_5a(bins5a, best_p_bg_5a[0], best_p_bg_5a[1], best_p_bg_5a[2], best_p_bg_5a[3])
  return 2 * np.sum(background + signal - events + xlogy(events, events/(background + signal)) )

def log_likelihood_5b(p, signal, Asimov): #Binned log likelihood for Poisson data
  background = fitting_function_5b(bins5b, p[0], p[1], p[2])
  if not(Asimov): events = events5b
  else: events = fitting_function_5b(bins5b, best_p_bg_5b[0], best_p_bg_5b[1], best_p_bg_5b[2])
  return 2 * np.sum(background + signal - events + xlogy(events, events/(background + signal)) )

def best_log_likelihood_5a(signal = np.zeros(65), Asimov = False): #Minimisation routine over background parameters

  res_list = []
  success_list = []

  for p in guesses5a:
    res = minimize(log_likelihood_5a, p, args=(signal, Asimov), tol = 0.01)
    if verbose > 2: print (p,res)
    res_list.append(res.fun)
    success_list.append(res.success)
    
  best_ll = min(res_list)
  
  success_rate = np.count_nonzero(success_list) / len(guesses5a)
  identical_results = np.count_nonzero(np.array(res_list) < best_ll + 0.01) / len(guesses5a)
  if verbose > 1:
    print('Success rate:', success_rate)
    print('Fraction of results within 0.01 of minimum:', identical_results)
  if success_rate == 0 or identical_results < 0.02: print('Warning! It looks like the minimisation has failed...')
  
  return best_ll

def best_log_likelihood_5b(signal = np.zeros(65), Asimov = False): #Minimisation routine over background parameters

  res_list = []
  success_list = []

  for p in guesses5b:
    res = minimize(log_likelihood_5b, p, args=(signal, Asimov), tol = 0.01)
    if verbose > 2: print (p,res)
    res_list.append(res.fun)
    success_list.append(res.success)
    
  best_ll = min(res_list)
  
  success_rate = np.count_nonzero(success_list) / len(guesses5b)
  identical_results = np.count_nonzero(np.array(res_list) < best_ll + 0.01) / len(guesses5b)
  if verbose > 1:
    print('Success rate:', success_rate)
    print('Fraction of results within 0.01 of minimum:', identical_results)
  if success_rate == 0 or identical_results < 0.02: print('Warning! It looks like the minimisation has failed...')
  
  return best_ll

background_log_likelihood_5a = best_log_likelihood_5a()
background_log_likelihood_5b = best_log_likelihood_5b()

def sqrt_safe(x):
  if x < 0: return 0
  else: return np.sqrt(x)
  
# For a given signal model, the following code calculates the 95% confidence level upper bound on the signal strength, i.e. it determines the value of mu such that
# -2 * (log L(mu * signal + background) - log L(background)) = 3.84
def signal_strength_bound(signal_high, signal_low, fix_params = False):

  if not(fix_params):
    
    def chi2(mu):
      result = best_log_likelihood_5a(signal = mu * signal_high) + best_log_likelihood_5b(signal = mu * signal_low)
      return result
          
  else:
  
    def chi2(mu):
      result = log_likelihood_5a(best_p_bg_5a, mu * signal_high, False) + log_likelihood_5b(best_p_bg_5b, mu * signal_low, False)
      return result

  initial_guess = 1

  cons = ({'type': 'ineq', 'fun': lambda x:  x})

  res = minimize(chi2, initial_guess, constraints=cons, tol=0.1)
 
  def test_statistic(mu):
      result = chi2(mu) - res.fun - 3.84
      return result
      
  while test_statistic(initial_guess) < 0: initial_guess *= 2  
  
  res2 = brentq(test_statistic, 0, initial_guess, xtol = 0.01)
  
  def CLs(mu):
  
    delta_chi2 = chi2(mu) - res.fun
    chi2_Asimov = best_log_likelihood_5a(signal = mu * signal_high, Asimov = True) + best_log_likelihood_5b(signal = mu * signal_low, Asimov = True)
        
    result = 0.05 - (1 - norm.cdf(sqrt_safe(delta_chi2)))/norm.cdf(sqrt_safe(chi2_Asimov) - sqrt_safe(delta_chi2))
        
    if verbose > 0: print('Trying signal strengh mu =',mu,'with result =',result)
    return result

  res3 = brentq(CLs, 0, initial_guess, xtol = 0.01)
  
  return res2, res3

for mzp in mzplist:
  mubound, mubound_CLs = signal_strength_bound(signal_dictionary_high[mzp], signal_dictionary_low[mzp])
  mubound_fixed, mubound_fixed_CLs = signal_strength_bound(signal_dictionary_high[mzp], signal_dictionary_low[mzp], fix_params = True)
  print('Bound on signal strength for M_Zp =',mzp,'GeV: ',mubound_CLs, '(', mubound_fixed, 'without background variation and without CLs)')
  print('Bound on cross section for M_Zp =',mzp,'GeV: ',mubound_CLs*cross_sections(mzp),'pb')

