import matplotlib.pyplot as plt
import numpy as np

from linear_fit_svd import linear_fit_svd
from scipy.optimize import curve_fit

"""
Simple testof a straight line without weights etc
"""

def lin( x, a, b ):
    return a + b * x

### test data
datalen = 53
fulllen = 150
xl = np.linspace( 3, 15, datalen )
xlf = np.linspace( 0, 15, fulllen )
yl = ( 5.11 + 2.71 * xl )
yln = yl + np.random.normal( size=datalen, scale=1.3 )


### cure_fit as reference
cfopt, cfcov = curve_fit( lambda x, a, b: a + b * x, xl, yln )
print ( cfopt )
print ( cfcov )
print( "---------------" )


### lin fit
lfr = linear_fit_svd(
    xl, yln,
    ( lambda x: np.ones( len(x)), lambda x: x )
)
sol = lfr.best_fit_parameters()
cov = lfr.covariance()
print( sol )
print( cov )
print( "solution and covarianmatrix should be identical" )
ylf = sol[0] + sol[1] * ( xlf )

sy2 = cov[0,0] * np.ones( fulllen ) + cov[1,1] * xlf**2 + 2 * cov[0,1] *xlf
sy = np.sqrt( sy2 )


### plotting
fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1 )
ax.scatter( xl, yln )
ax.plot( xlf, ylf )
ax.plot( xlf, ylf + 3 * sy )
ax.plot( xlf, ylf - 3 * sy )
plt.show()

