import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions( linewidth=250, precision=5 )

from linear_fit_qr import linear_fit_qr
from scipy.optimize import curve_fit

"""
Simple testof a straight line with given covariance matrix
"""

def lin( x, a, b ):
    return a + b * x
### test data
datalen = 8
slist = 5 * np.random.random( size=datalen )
print( slist )
fulllen = 150
xl = np.linspace( 3, 15, datalen )
xlf = np.linspace( 0, 15, fulllen )
yl = ( 5.11 + 2.71 * xl )


#create n * n matrix
w = np.random.normal( size=datalen**2, scale=0.01 )
w = w.reshape( ( datalen, datalen ) )
# make it symmetric
w = 0.5 * ( w + np.transpose( w ) )
#remove the diagonal
d = np.diag( w )
d = np.diag( d )
w = w - d
### and replace it with the "correct" positive values
d = 1 / slist**2
w = w + np.diag( d )
v = np.linalg.inv( w )
### choose the off diagonal elements sufficiently small to ensure
### positive definite
### without actually putting the correlation into the data
yln = yl + np.fromiter( ( np.random.normal( scale=s ) for s in slist ), float)


### cure_fit as reference
cfopt, cfcov = curve_fit( 
    lambda x, a, b: a + b * x,
    xl, yln,
    sigma=v,
    absolute_sigma=True
)
print ( cfopt )
print ( cfcov )
print( "---------------" )


### lin fit
lfr = linear_fit_qr(
    xl, yln,
    ( lambda x: np.ones( len(x)), lambda x: x ),
    w = w
)
sol = lfr.bestfitparameters()
cov = lfr.covariance()
print( sol )
print( cov )
print( "solution and covarianmatrix should be identical" )
ylf = sol[0] + sol[1] * ( xlf )

sy2 = (
    cov[ 0, 0 ] * np.ones( fulllen ) +
    cov[ 1, 1 ] * xlf**2 +
    cov[ 0, 1 ] * xlf * 2
)
sy = np.sqrt( sy2 )


### plotting
fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1 )
ax.errorbar( xl, yln, yerr = slist, ls='', marker='o' )
ax.plot( xlf, ylf )
ax.plot( xlf, ylf + 3 * sy )
ax.plot( xlf, ylf - 3 * sy )
plt.show()

