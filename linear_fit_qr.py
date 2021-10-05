import numpy as np
from scipy.linalg import qr
from scipy.linalg import svd
from scipy.linalg import solve_triangular
from warnings import warn


"""
Note, in therory this might work with complex data, but then
most of the transpose operations on matrices would also requite a
complex conjugate
"""

from LinearFitResult import LinearFitResult
from checkinput import _check_weighting
from checkinput import _check_data
from checkinput import _check_functions


def linear_fit_qr( xdata, ydata, funclist, w=None ):
    """
    Linear fit procedure using qr decomposition. Calculates the best
    linear combination of the function, evaluated on xdata, to
    approximate ydata. Minimizes the weightes least sqare.Weights may
    be specified. Uses the fact that ( S x - y ).T W ( S x - y ) can
    be decomposed into ( S x - y ).T F.T F ( S x - y ) and therefore
    be remapped to the unweighted problem ( F S x - F y ) = ( A x - z )
    Eventually, it is tested against scipy.optimize.curve_fit. The
    point is to show that there is no magic in linear fits even with a
    given covariance- or precision matrix. It is really straight
    forward and can be done with simple tools from linear algebra and
    one or two little tweaks to avoid numerical instability.

    :param xdata: 
    :type xdata: list, tuple or ndarray of int or float
    :param ydata: 
    type ydata: list, tuple or ndarray of int or float
    @param funclist: list, tuple or ndarray of single valued callables

    Keyword args:
        :w: weighting.This is used in the sense of an absolute_sigma,
        see scipy.optimize.curve_fit
        :type w: None, list, tuple or ndarray of int or float

    :return: The fit results
    :rtype: LinearFitResult

    :raises TypeError: if xdata or ydata are not iterables of type int or float
    :raises TypeError: if xdata or ydata are not iterables

    """
    n = _check_functions( funclist )
    m = _check_data( n, xdata, ydata )

    ###  weighting
    fmx = _check_weighting( m, w=w )
    ###################
    ### calc best fit
    ###################
    lfr = LinearFitResult()
    lfr._degreesoffreedom = m - n
    ST = np.array( [ f( xdata ) for f in funclist ] )           ## only used to calc S
    S = np.transpose( ST )
    FS = np.dot( fmx, S )                                        ## re-map due to weighting
    fy = np.dot( fmx, ydata )                                   ## re-map due to weighting
    q, r = qr( FS )                                              ## decomposition
    rred = r[ : n ]                                             ## making it square...skip the zeros
    qfy = np.dot( np.transpose( q ), fy )
    qfyred = qfy[ : n ]
    epsilonred = qfy[ n : ]
    ### note the part np.dot(...)[ n:] corresponds to the
    ### orthoganal transformed weighted residuals, i.e. without weigthing
    ### its norm is equal to the norm of the residuals
    lfr._bestfitparameters = solve_triangular( rred, qfyred )
    lfr._functionvalues = np.dot( S, lfr.best_fit_parameters() )  ## by definition
    lfr._residuals = ydata - lfr.function_values()
    ### the following s2 fails if n>=m
    lfr._totalerror = np.sqrt(
        np.dot(
            lfr.residuals(),
            lfr.residuals()
        ) / lfr.degrees_of_freedom()
    )
    lfr._totalweightederror = np.sqrt(
        np.dot(
            epsilonred,
            epsilonred
        )
    )
    lfr._precision = np.dot( np.transpose( r ), r )
    ####################
    ### So W = R.T R => if s are the singular values (note, not the eigenvalues)
    ### of R then W has eigenvalues s^2. So if all singular values 
    ### are non-zero, W is positive definite and invertable
    ### https://math.stackexchange.com/a/3291423/233820
    ####################
    try:
        lfr._covariance =  np.linalg.inv( lfr.precision() )
        if w is None:
            lfr._covariance *= lfr.total_error()**2
    except np.linalg.LinAlgError as msg:
        warn( 
            "coavriance matrix could not be calculated due to {}".format( msg ),
            UserWarning
        )
    return lfr
