import matplotlib.pyplot as plt
import numpy as np
from linear_fit_svd import linear_fit_svd

"""
fitting a sine with polynomials of very high order
reduce numerical issues by first scaling into the [-1,1]
interval.The result is scaled back to the original interval
"""


def make_func_list( order ):
    """
    wrapper to make a set of polynomials to be fitted
    """
    out = list()
    for i in range( order + 1 ):
        out.append( lambda x, i=i: x**i )
    return out

### data
xl = np.linspace( -2, 8, 55 )
xf = np.linspace( -2, 8, 255 )
xmax = max( np.abs( xl ) )
### scale x to be in the range [-1,1] or less
xls = xl / xmax
yl = np.fromiter( ## test function to be fitted
    ( 2.1 * np.sin( 2.83 * x ) for x in xl),
    float
)


### plotting and fitting
fig = plt.figure( figsize=( 8, 6) )
ax = fig.add_subplot( 1,1, 1 )

ax.plot( xl, yl, ls='', marker='o' )

for order in[ 5 , 10, 15, 20 ]: ## fitting for different orders
    fl  = make_func_list( order )
    lfr = linear_fit_svd( xls, yl, fl )
    opt, cov = lfr.best_fit_parameters(), lfr.covariance()
    fit = np.array( [ o * f( xls ) for o,f in zip( opt, fl ) ] )
    fit = np.sum( fit, axis=0 )
    ### backscaled
    fits = np.array(
        [ o / ( xmax**n ) * f( xf ) for n, o, f in zip(
            range( order + 1 ), opt, fl
        ) ]
    )
    fits = np.sum( fits, axis=0 )
    ax.plot( xf, fits )
plt.tight_layout()
plt.show()