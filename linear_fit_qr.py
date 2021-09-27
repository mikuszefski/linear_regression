import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.linalg import solve_triangular


def make_func_list( order ):
    """
    wrapper to make a set of polynomials to be fitted
    """
    out = list()
    for i in range( order + 1 ):
        out.append( lambda x, i=i: x**i )
    return out


def my_linear_fit( xdata, ydata, funclist, w=None ):
    """
    fitting procedure using qr decomposition
    """
    n = len( funclist )
    ST = np.array( [ f( xdata ) for f in funclist ] )
    S = np.transpose( ST )
    q, r = qr( S )
    rred = r[ : n ] ### making it square...skip the zeros
    yred = np.dot( np.transpose( q ), ydata )[ : n ]
    sol = solve_triangular( rred, yred )
    yth = np.dot( S, sol )
    diff = ydata - yth
    s2 = np.sum( diff**2 ) / ( len( diff ) - len( funclist ) )
    cov =  np.linalg.inv( np.dot( np.transpose( r ), r ) ) * s2
    return sol, cov

if __name__ == "__main__":
    ### data
    xl = np.linspace( -2, 8, 55 )
    xmax = max( np.abs( xl ) )
    ### scale x to be in the range [-1,1] or less
    xls = xl / xmax
    yl = np.fromiter(
        ( 2.1 * np.sin( 2.83 * x ) for x in xl),
        float
    )

    ### plotting
    fig = plt.figure( figsize=( 10, 2) )
    ax = fig.add_subplot( 1, 2, 1 )
    bx = fig.add_subplot( 1, 2, 2 )
    
    ax.plot( xl, yl, ls='', marker='o' )
    bx.plot( xls, yl, ls='', marker='o' )

    for order in[ 5 , 10, 15, 20 ]: ## fitting for different orders
        fl  = make_func_list( order )
        opt, cov = my_linear_fit( xls, yl, fl )
        fit = np.array( [ o * f( xls ) for o,f in zip( opt, fl ) ] )
        fit = np.sum( fit, axis=0 )
        ### backscaled
        fits = np.array( [ o/(xmax**n) * f( xl ) for n, o, f in zip( range(order + 1), opt, fl ) ] )
        fits = np.sum( fits, axis=0 )
        ax.plot( xl, fits)
        bx.plot( xls, fit)
    plt.tight_layout()
    # ~plt.savefig( "sine.png" )
    plt.show()



