
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


def linear_fit_svd( xdata, ydata, funclist, w=None ):
    """
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
    FS = np.dot( fmx, S )                                       ## re-map due to weighting
    yf = np.dot( fmx, ydata )                                   ## re-map due to weighting
    ###now the reduced problem is min( FS a - yf )
    u, sig, vh = svd( FS )                                       ## decomposition
    yfu = np.dot( np.transpose( u ), yf )
    ### now with z = vh a we solve sig z  = yfu
    sigmared = np.diag( sig )
    omegared = np.diag( 1 / sig )                               ## it's diagonal, so just invert it
    yfured = yfu[ :n ]
    epsilonwred = yfu[ n: ]
    lfr._bestfitparameters = np.dot(                            ## the inverse
        np.transpose( vh ),
        np.dot( omegared, yfured )
    )
    lfr._functionvalues = np.dot( S, lfr.best_fit_parameters() )  ## by definition
    lfr._residuals = ydata - lfr.function_values()
    ### the following _totalerror fails if n>=m
    lfr._totalerror = np.sqrt(
        np.dot(
            lfr.residuals(),
            lfr.residuals()
        ) / lfr.degrees_of_freedom()
    )
    lfr._totalweightederror = np.sqrt(
        np.dot( epsilonwred, epsilonwred )
    )
    lfr._covariance = np.dot(
        np.transpose( vh ),
        np.dot(
            omegared**2,
            vh
        )
    )
    lfr._precision = np.dot(
        np.transpose( vh ),
        np.dot(
            sigmared**2,
            vh
        )
    )
    if w is None:
        lfr._covariance *= lfr.total_error()**2
    return lfr
