# linear_regression
using basic python tools to implement linear regression

**Introduction**

Working in the curve- and data-fitting section on stack-exchange, I see 
plenty of question on Python functions making a non linear-fit, when 
the post actually just requires a linear optimization. Of course one 
can get the answer, but most of the time, it is unnecessary and 
obscuring the true nature. To provide some insight and according 
Python tools and methods, I wrote down the following paragraphs.

**The linear fit problem**

Linear regression means, we have to find the linear coefficients of 
functions to get a linear combination that best fits some data:

y = a0 + a1 f1(x ) + a2 f2( x ) + ... + an fn( x )

and we have data y .T= ( y1, ..., ym ), T denoting the transposed,
for positions x.T = ( x1, ..., xm ) with m > n, i.e the system is 
over-determined. With a.T = ( a0, ..., an ), the equation can be 
written as

y = M a,

where

M = ( 1, f1, ..., fn) and fn.T = ( fn( x1 ), ..., fn( xm ) )

**Minimisation**

Usually there is no exact solution and y - M a = ε with residuals ε 
and we want to minimize ε.T ε, which means minimizing 
( y - M a ).T ( y - M a ). If the measured values have different error, 
this equation might get a weighting, via a weighting matrix W, such 
that it becomes min_a { ( y - M a ).T W ( y - M a ) }

**Diagonalisation and weighted data**

Before solving this problem, it is better to make clear that a 
weighting does not change the original problem. The weighting matrix 
is the inverse of the covariance matrix V of the y-values, 
i.e. W = V⁻¹. So assuming that the covariance matrix has an inverse 
and knowing that it is symmetric, i.e. V.T = V, we know that W can be 
decomposed into W = O.T D O, where O is an orthogonal matrix,
i.e. O.T = O⁻¹, and D is diagonal. As D is diagonal one obviously has
D = D.T and there is a matrix C with C = C.T and C.T C = D.,
i.e. W = O.T C.T C O = F.T F. One can write, hence,

( y - Ma ).T F.T F ( y - M a ) 
    = ( F y - F M a ).T ( F y - F M a )
    = ( η - N a ).T ( η - N a )

The weighted problem can, therefore, be mapped on an unweighted problem
with measurements η and function matrix N.

**Naive solution**

Expanding the above minimization problem one gets

η.T η - η.T N a - a.T N.T η + a.T N.T N a

Minimizing, i.e. setting the derivative with respect to a equal to 
zero results in

N.T η = N.T Na,

which means

a = (N.T N)⁻¹ N.T η

**Improving by matrix decomposition**

While this is mathematically correct, very often it is numerically 
unstable. Instead of calculating the inverse it is better to solve the
equation step-by-step. To do so, one starts by making a
QR-decomposition. One way would be the QR-decomposition of N.T N,
but it can actually be done for non-square matrices as well. It is
better to make it for N = Q R, with Q being a m ⨯ m orthogonal matrix
and R being a m ⨯ n upper triangular matrix, i.e. R has m - n rows of
zeros at the bottom.

In Python we get this decomposition simply by

Q, R = scipy.linalg.qr( N )

Now we have R a =Q.T η

Due to all the zeros in R this equation cannot be true. The lower
entries somewhat represent the error that we cannot remove by
optimizing a. We can however match the upper part. In Python we remove
the lower part and use the method to solve triangular systems, which
avoids matrix inversion. This has the simple form

a = scipy.linalg.solve_triangular(
    R[ : n ], 
    numpy.dot( numpy.transpose( Q ), η )[ : n ]
)

**Errors and error propagation.**

The errors of a are propagating from y. In a general case one may have

a = f( y ), which may be linear in the form a = H y

If y has a covariance matrix V with elements Vij then one might ask
what the covariance matrix Z with elements Zij of a looks like.
It is given by standard error propagation, namely

Zij = dai / dyk ⨯ daj / dyl ⨯ Vkl = Jik Jjl Vkl = Jik Vkl (J.T)lj

so Z = J V J.T

where J is the Jacobian of f( y ). In the linear case J = H.

In the above case we have H = R⁻¹ Q.T. In case that we do not know
the errors of y, we estimate them from our results, via the
residuals ε, assuming that

V ≈ ε.T ε /( m - n ) 1 = s² 1

where 1 is the identity matrix. In this case J V J.T becomes

(R⁻¹ Q.T) s² 1 (R⁻¹ Q.T).T = s² R⁻¹ Q.T Q (R.T)⁻¹ = s² (R.T R)⁻¹
Propagation in the case of weighted data

Let's make a short and hopefully not too confusing detour to check the
results in case of a weighted problem. We already established that the
problem is written as

F y = F M a

or in the somewhat naive approach that constructs square matrices and
allows for simple matrix inversion (at least in theory)

M.T F y = M.T F M a

and therefore

a = ( M.T F M )⁻¹ M.T F y

i.e. J = ( M.T F M )⁻¹ M.T F

and

J V J.T = ( M.T F M )⁻¹ M.T F V ( ( M.T F M )⁻¹ M.T F ).T

= ( M.T F M )⁻¹ M.T F V F.T M ( ( M.T F M ). T )⁻¹

= ( M.T F M )⁻¹ M.T F V F.T M ( M.T F.T M )⁻¹

= ( M.T F M )⁻¹ M.T F V F.T F F⁻¹ M ( M.T F.T M )⁻¹

= ( M.T F M )⁻¹ M.T F V W F⁻¹ M ( M.T F.T M )⁻¹

= ( M.T F M )⁻¹ M.T F V V⁻¹ F⁻¹ M ( M.T F.T M )⁻¹

= ( M.T F M )⁻¹ M.T F F⁻¹ M ( M.T F.T M )⁻¹

= ( M.T F M )⁻¹ M.T M ( M.T F.T M )⁻¹

= M⁻¹ F⁻¹ ( M.T )⁻¹ M.T M M⁻¹ ( F.T )⁻¹ ( M.T )⁻¹

= M⁻¹ F⁻¹ ( F.T )⁻¹ ( M.T )⁻¹

= M⁻¹ ( F.T F )⁻¹ ( M.T )⁻¹

= M⁻¹ W⁻¹ ( M.T )⁻¹

=( M.T W M )⁻¹ = ( ( F M ).T F M )⁻¹

Now considering the QR-decomposition with F M = N = Q R, we have

( ( F M ).T F M )⁻¹ = ( ( Q R ).T Q R )⁻¹ = ( R.T R )⁻¹

which is the non-weighted result only without the s². This is logic
as the uncertainty is known and already incorporated via F.
