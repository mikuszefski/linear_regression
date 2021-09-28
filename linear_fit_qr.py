import numpy as np
from scipy.linalg import qr
from scipy.linalg import solve_triangular
from warnings import warn

class LinearFitResult( object ):
    """
    Simple container class to hold fit results.
    Provides a few self explaining getter functions.
    """

    def __init__( self ):
        self._covariance = None
        self._errorestimate = None
        self._bestfitparameters = None
        self._residuals = None
        self._functionvalues = None
        self._degreesoffreedom = None
        self._precision = None

    def errorestimate( self ):
        return self._errorestimate

    def bestfitparameters( self ):
        return self._bestfitparameters

    def residuals( self ):
        return self._residuals

    def functionvalues( self ):
        return self._functionvalues

    def degreesoffreedom( self ):
        return self._degreesoffreedom

    def precision( self ):
        return self._precision

    def covariance( self ):
        return self._covariance


def linear_fit_qr( xdata, ydata, funclist, w=None ):
    """
    Linear fit procedure using qr decomposition. Calculates the best
    linear combination of the function, evaluated on xdata, to approximate
    ydata. Minimizes the weightes least sqare.Weights may be specified.
    Uses the fact that (S x - y ).T W (S x - y ) can be decomposed
    into (S x - y ).T F.T F (S x - y ) and therefore be remapped
    to the unweighted problem ( F S x - F y ) = (A x - z)
    Eventually, it is tested against scipy.optimize.curve_fit. The point
    is to show that there is no magic in linear fits even with a given
    covariance- or precision matrix. It is really straight forward and can be done
    with simple tools from linear algebra and one or two little tweaks
    to avoid numerical instability.

    :param xdata: 
    :type xdata: list, tuple or ndarray of int or float (maybe complex)
    :param ydata: 
    type ydata: list, tuple or ndarray of int or float (maybe complex)
    @param funclist: list, tuple or ndarray of single valued callables

    Keyword args:
        :w: weighting.This is used in the sense of an absolute_sigma,
        see scipy.optimize.curve_fit
        :type w: None, list, tuple or ndarray of int or float

    :return: The fit results
    :rtype: LinearFitResult

    :raises TypeError: if xdata or ydata are not iterables of type int or float
    :raises ValueError: if xdata and ydata have diffeerent length
    :raises ValueError: if length of ydata is less or equal than length of funclist
    :raises ValueError: if w is not list, tuple or array of positive int or float
    :raises ValueError: if w is not positive definite matrix of int or float
    :raises TypeError: if w is not of shape (m,) or (m,m), where m is the length of ydata
    :raises TypeError: if w is a matrix of shape (m,m), but not symmetric
    """
    n = len( funclist )
    m = len( xdata )
    ###################
    ### the following block
    ### only checks for 
    ### correct input
    ###################

    ### proper data types
    if not np.fromiter(
                (
                    isinstance( x, (int, float) ) for x in xdata
                ), bool
    ).all():
        raise TypeError ( "xdata is neither int nor float" )
    if not np.fromiter(
                (
                    isinstance( x, (int, float) ) for x in ydata
                ), bool
    ).all():
        raise TypeError ( "ydata is neither int nor float" )

    ### proper data length
    if m != len( ydata ):
        raise ValueError ( "x- and y-data  of unequal length." )
    if m <= n:
        raise ValueError ( "Exact or under determined problem." )
        ###m = n could be solved, but one cannot estimate an s^2

    ###  weighting
    if w is None:
        fmx = np.identity( m )
    else:
        fmx2 = np.asarray( w )
        if fmx2.shape == ( m, ):                            ## is vector
            ### propertype of elements in w
            if (
                np.fromiter(
                    (isinstance( elem, (int, float) ) for elem in fmx2),
                    bool
                ).all()
            ):
                ### must be positive definite
                if np.fromiter(
                    ( elem > 0 for elem in fmx2 ),
                    bool
                ).all():
                    fmx = np.diag( np.sqrt( fmx2 ) )
                else:
                    raise ValueError ("Weighting is not a list of positive int or float")
                    
            else:
                raise TypeError ("Weighting is not of type int or float")
        elif fmx2.shape == ( m, m ):
            if np.fromiter(
                (
                    isinstance( elem, (int, float) ) for elem in np.concatenate( fmx2 ) 
                ), bool
            ).all():
                if not np.allclose( fmx2, np.transpose( fmx2 ) ):
                    raise TypeError ("covariance matrix is not symmetric.")
                evals, evecs = np.linalg.eig( fmx2 )
                if not np.fromiter(
                    ( elem > 0 for elem in evals),           ## must be positive definite
                    bool
                ).all():
                    raise ValueError (" wighting matrix is not positive definite")
                ### here D = diag evals
                ### and O = evacsT
                ### then fmx = do.T do
                ###with
                c = np.diag( np.sqrt( evals ) )             ## c.T c = d
                o = np.transpose( evecs )
                fmx = np.dot( c, o )
            else:
                raise TypeError ("Weighting is not a matrix of int or float")
        else:
            raise TypeError (
                "w is neither of shape ({m},) nor ({m},{m})".format( m=m )
            )
    ###################
    ### calc best fit
    ###################
    lfr = LinearFitResult()
    lfr._degreesoffreedom = m - n
    ST = np.array( [ f( xdata ) for f in funclist ] )       ## only used to calc S
    S = np.transpose( ST )
    S = np.dot( fmx, S )                                    ## re-map due to weighting
    yf = np.dot( fmx, ydata )                                ## re-map due to weighting
    q, r = qr( S )                                          ## decomposition
    rred = r[ : n ]                                         ## making it square...skip the zeros
    yred = np.dot( np.transpose( q ), yf )[ : n ]
    lfr._bestfitparameters = solve_triangular( rred, yred )
    lfr._functionvalues = np.dot( S, lfr.bestfitparameters() ) ## by definition
    lfr._residuals = ydata - lfr.functionvalues()
    ### the following s2 fails if n>=m
    s2 = np.dot( lfr.residuals(), lfr.residuals() ) / lfr.degreesoffreedom()
    lfr._errorestimate = np.sqrt( s2 )
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
            lfr._covariance *= s2
    except np.linalg.LinAlgError as msg:
        warn( 
            "coavriance matrix could not be calculated due to {}".format( msg ),
            UserWarning
        )
    return lfr
