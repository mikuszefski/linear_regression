
class LinearFitResult( object ):
    """
    Simple container class to hold fit results.
    Provides a few self explaining getter functions.
    """

    def __init__( self ):
        self._covariance = None
        self._bestfitparameters = None
        self._residuals = None
        self._totalerror= None
        self._totalweightederror = None
        self._functionvalues = None
        self._degreesoffreedom = None
        self._precision = None



    def best_fit_parameters( self ):
        return self._bestfitparameters

    def residuals( self ):
        return self._residuals

    def total_error( self ):
        return self._totalerror

    def total_weighted_error( self ):
        return self._totalweightederror

    def function_values( self ):
        return self._functionvalues

    def degrees_of_freedom( self ):
        return self._degreesoffreedom

    def precision( self ):
        return self._precision

    def covariance( self ):
        return self._covariance
