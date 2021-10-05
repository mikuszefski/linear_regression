import numpy as np

def _check_weighting( m, w=None ):
    """
    :param m:length of data
    :type m: int

    :return fmx: square root ofweighting matrix W, ie. fmx.T fmx = W.
    :return type: ndarray of shape (m, m)

    :raises TypeError: if w is not list, tuple or array of int or float
    :raises TypeError: if w is not of shape (m,) or (m,m), where m is the length of ydata
    :raises TypeError: if w is a matrix of shape (m,m), but not symmetric
    :raises ValueError: if length of ydata is less or equal than length of funclist
    :raises ValueError: if xdata and ydata have diffeerent length
    :raises ValueError: if w is a matrix of shape (m,m), but not positive definite."
    :raises ValueError: if w is an iterable of shape (m,)with entries less or equal zero"
    """
    if w is None:
        fmx = np.identity( m )
    else:
        fmx2 = np.asarray( w )
        ### proper type of elements in w
        if not np.fromiter(
            (
                isinstance( 
                    elem, (int, float) 
                ) for elem in fmx2.flat
            ), bool
        ).all():
            raise TypeError ("Weighting is not of type int or float")
        if fmx2.shape == ( m, ):                                ## is vector
            ### must be positive definite
            if np.fromiter(
                ( elem > 0 for elem in fmx2 ),
                bool
            ).all():
                fmx = np.diag( np.sqrt( fmx2 ) )
            else:
                raise ValueError(
                    "Weighting is not a list of positive int or float"
                )
        elif fmx2.shape == ( m, m ):                            ## is matrix
            ### symmetry
            if not np.allclose( fmx2, np.transpose( fmx2 ) ):
                raise TypeError ("covariance matrix is not symmetric.")
            evals, evecs = np.linalg.eig( fmx2 )
            ### positive definite
            if not np.fromiter(
                ( elem > 0 for elem in evals),
                bool
            ).all():
                raise ValueError(
                    " wighting matrix is not positive definite"
                )
            ### here D = diag evals
            ### and O = evacsT
            ### then W = o.T d o
            ###with
            c = np.diag( np.sqrt( evals ) )                 ## c.T c = d
            o = np.transpose( evecs )
            fmx = np.dot( c, o )                            ## fmx.T fmx = W
        else:
            raise TypeError (
                "w is neither of shape ({m},) nor ({m},{m})".format( m=m )
            )
    return fmx

def _check_functions( functionlist ):
    """
    check if input is an iterableof callables
    :param functionlist:
    :type funtionlist: list, tuple or ndarray
    
    :returns: number of callable functions
    :return type: int

    :raises TypeError: if input is not iterable type
    :raises TypeError: elements of input are not callable
     
    """
    if not isinstance( functionlist, ( list, tuple, np.ndarray ) ):
        raise TypeError(
            "List of functions must be type, list, tuple or array"
        )
    if not np.fromiter(
        (
            callable( obj ) for obj in functionlist
        ),
        bool
    ).all():
        raise TypeError( "Non-callableelements in function list ")
    return len( functionlist )

def _check_data( n, xdata, ydata ):
    """
    check for consisten data
    :param n: number of independents against data is fitted
    :type n: int
    :param xdata:
    :type xdata: tuple, list or ndarray of int float
    :param ydata:
    :type ydata: tuple, list or ndarray of int float

    :returns: number of data entries
    :return type: int
    
    :raises TypeError: if input types or neither list, tuple not ndarrays
    :raises:
    """
    ### proper data types
    if not isinstance( xdata, ( tuple, list, np.ndarray ) ):
        raise TypeError ( "xdata must be 1D iterable" )
    if not isinstance( ydata, ( tuple, list, np.ndarray ) ):
        raise TypeError ( "ydata must be 1D iterable" )
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
    m = len( xdata )
    if m != len( ydata ):
        raise ValueError ( "x- and y-data of unequal length." )
    if m <= n:
        raise ValueError ( "Exact or under determined problem." )
        ###m = n could be solved, but one cannot estimate an s^2
    return m
