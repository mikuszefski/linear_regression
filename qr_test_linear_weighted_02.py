import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions( linewidth=250, precision=5 )
from scipy.linalg import qr

from linear_fit_qr import linear_fit_qr
from scipy.optimize import curve_fit
"""
Simple testof a straight line with given covariance matrix
"""

def lin( x, a, b ):
    return a + b * x
### test data
datalen = 8
fulllen = 150
xl = np.linspace( 3, 15, datalen )
xlf = np.linspace( 0, 15, fulllen )
yl = ( 5.11 + 2.71 * xl )


######################
### now creating a covariance matrix for the data
######################

###it must be positive so diagonals are
ss = np.diag( 25 * np.random.random( size=datalen ) )
### now we need a random orthogonal transformation
### and for correlated data we need C such that C C.T = Vd
q, r = qr( np.random.random( size=datalen**2).reshape( (datalen, datalen) ) )
C = np.dot( q, np.sqrt( ss ) )
Vd = np.dot( C, np.transpose( C ) )
## now Vd is positive definite and symmetric
W = np.linalg.inv( Vd )
# ~yln = np.random.multivariate_normal( yl, Vd, size=1)[0]
yln = yl + np.dot(
    C,
    np.random.normal(
        size=datalen,
    )
)


### cure_fit as reference
cfopt, cfcov = curve_fit( 
    lambda x, a, b: a + b * x,
    xl, yln,
    sigma=Vd,
    absolute_sigma=True
)
print ( cfopt )
print ( cfcov )
print( "---------------" )


### lin fit
lfr = linear_fit_qr(
    xl, yln,
    ( lambda x: np.ones( len(x)), lambda x: x ),
    w = W
)
sol = lfr.best_fit_parameters()
cov = lfr.covariance()
print( sol )
print( cov )
print( "solution and covarianmatrix should be identical" )

### prepare data to plot
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
ax.errorbar( xl, yln, yerr = np.sqrt( np.diag( Vd ) ), ls='', marker='o' )
ax.plot( xlf, ylf )
ax.plot( xlf, ylf + 3 * sy )
ax.plot( xlf, ylf - 3 * sy )
ax.plot( xl, yl, ls=':' )
plt.show()

