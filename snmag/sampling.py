import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys

data = pd.read_csv('single_SN_dist_Linder.txt', sep=" ", header=None)
x = data[0]
y = data[1]

y = y / y.sum()

def interpol(x,y, numNew):
    """
    interpolate
    """
    x_new = np.linspace(min(x),max(y),num=numNew)
    y_new = np.interp(x_new,x,y)

    return x_new, y_new


def sample(x,y, numSamples):
      """
      gives numSamples samples from the distribution funciton fail
      parameters
      """
      y /= y.sum()
      return np.random.choice(x, size=numSamples, replace=True, p=y)

run_samples = sample(x,y, 1000000)
print(len(np.unique(run_samples)))
print(len(x))
sys.exit(1)
# Compute a histogram of the sample
#bins = np.linspace(-5, 5, 30)
bins = x
histogram, bins = np.histogram(samples, bins=bins, normed=True)
bin_centers = 0.5*(bins[1:] + bins[:-1])
cdf = np.cumsum(y)

plt.figure(figsize=(6, 4))
#plt.plot(bin_centers, histogram, label="Histogram of samples")
plt.plot(bins,cdf, label ='CDF')
#plt.plot(bin_centers, pdf, label="PDF")
plt.legend()
plt.show()




#np.random.choice(bin, centers,p=normalized hist) # gives you coarse magnification

"""
# Load data from statsmodels datasets
#data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
data = pd.read_csv('single_SN_dist_Linder.txt', sep=" ", header=None)
# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=len(data), normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
#ax = data.plot()
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(u'All Fitted Distributions')
ax.set_xlabel(u'magnification')
ax.set_ylabel('PDF')

# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=len(data), normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Temp. (°C)')
ax.set_ylabel('Frequency')
plt.show()
"""
