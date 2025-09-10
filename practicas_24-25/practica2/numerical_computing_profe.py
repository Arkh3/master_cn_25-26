# -*- coding: utf-8 -*-
"""
Utilities for the course on Numerical Computing 
(Master's program in Data Science)

Created on Sun 2024-09-22

@author: alberto.suarez@uam.es
"""


import math
import warnings
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable


def numerical_derivative(
    f: Callable[[np.ndarray], np.ndarray],
    x0: float,
    h: float = 1.0e-6,
    scale_h: bool = True,
) -> float:
    """ Estimate of the derivative by divided (central) differences.

    Args:
        f: Function whose derivative we wish to determine.
        x0: Point at which the derivative is computed.
        h: Increment.

    Returns:
        A numerical estimate of the derivative.
        
    Examples:
       
        >>> 1.0 - numerical_derivative(np.exp, 0.0, scale_h=False)
        2.6755486715046572e-11
        
        >>> numerical_derivative(np.exp, np.array([1.0, 2.0]))
        array([2.71828183, 7.3890561 ])

        >>> x = np.reshape(np.logspace(-101, 101, 6), (2, 3))
        >>> f = np.sqrt
        >>> df_dx = lambda x: 0.5 / np.sqrt(x) 
        >>> 1.0 - numerical_derivative(f, x) / df_dx(x) # relative error
        array([[-8.37323544e-11,  8.13707990e-11,  2.04004591e-11],
               [-1.92157401e-12,  2.61047850e-11,  9.60679314e-11]])

    """
    

def numerical_second_derivative(
    f: Callable[[np.ndarray], np.ndarray],
    x0: float,
    h: float = 1.0e-4,
    scale_h: bool = True,
) -> float:
    """ Estimate of the second derivative by divided differences.

    Args:
        f: Function whose derivative we wish to determine.
        x0: Point at which the derivative is computed.
        h: Increment.

    Returns:
        A numerical estimate of the derivative.
        
    Examples:
       
        >>> 1.0 - numerical_second_derivative(np.exp, 0.0, scale_h=False)
        -5.024759275329416e-09
        
        >>> numerical_second_derivative(np.exp, np.array([1.0, 2.0]))
        array([2.71828187, 7.3890561 ])

        >>> x = np.reshape(np.logspace(-101, 101, 6), (2, 3))
        >>> f = np.sqrt
        >>> d2f_dx2 = lambda x: -0.25 / x / np.sqrt(x) 
        >>> 1.0 - numerical_second_derivative(f, x) / d2f_dx2(x) # relative error
        array([[-5.31950879e-08,  3.18853937e-08, -2.63389912e-08],
               [ 9.54296209e-09,  7.32422989e-08,  1.94053817e-08]])
    """
    

def newton_raphson(
    f: Callable,
    df_dx: Callable,
    seed: float,
    tol_abs: float = 1.0e-6,
    tol_rel: float = None,
    max_iters: int = 50
) -> float:
    """ Zero of the function :math:`f` using the Newton-Raphson method.

    Args:
        f: Function whose zero we wish to determine.
        df_dx: Derivative of :math:`f`.
        seed: Initial estimate of the zero (close to an actual one).
        tol_abs: Absolute error target.
        tol_rel: Relative error target.
        max_iters: Maximum number of iterations.

    Returns:
        An approximation of a zero of :math:`f` and its estimated error.

    Examples:
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> df_dx = lambda x: -np.sin(x) - 0.1
        >>> seed = 0.1
        >>> f_zero, error = newton_raphson(f, df_dx, seed, max_iters=20)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = 5.267 (1.1e-09),   f(f_zero) = -2.22e-16.
        
     
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> df_dx = lambda x: -np.sin(x) - 0.1
        >>> x_0 = np.pi
        >>> f_zero, error = newton_raphson(f, df_dx, x_0, max_iters=20)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = -9.679 (2.8e-09),   f(f_zero) = 3.33e-16.
    
        >>> f = lambda x: np.cos(x)
        >>> df_dx = lambda x: -np.sin(x)
        >>> seed = 0.0
        >>> f_zero, error = newton_raphson(f, df_dx, seed, max_iters=20)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = nan (nan),   f(f_zero) = nan.
        
    """
    if tol_abs is None:
        tol_abs = np.finfo(float).tiny

    if tol_rel is None:
        tol_rel - np.finfo(float).tiny

    iter = 0
    x = seed

    if (df_dx(x) == 0.0):
        warnings.warn('newton raphson no converge con la derivada = 0')
        return np.nan, np.nan

    delta_x = f(x)/df_dx(x)

    while (np.abs(delta_x) > tol_abs and (x == 0 or np.abs(delta_x/x) > tol_rel) and iter < max_iters):
        iter += 1

        if(df_dx(x) == 0.0):
            warnings.warn('newton raphson no converge con la derivada = 0')
            return np.nan, np.nan

        delta_x = f(x)/df_dx(x)
        x -= delta_x

    if iter == max_iters:
        warnings.warn('newton raphson no converge con la derivada = 0')

    return x, np.abs(delta_x)


#################################################################33
# Newton raphson vectorial

N_vals
###################################################################

def bisection(
    f: Callable,
    x_low: float,
    x_up: float,
    tol_abs: float = 1.0e-6,
    tol_rel: float = None,
    max_iters: int = 100
) -> float:
    """ Zero of the function :math:`f` using the bisection method.

    Find a zero in the interval :math:`\left( x_{low}, x_{up} \right)` 
    assuming that :math:`f(x_low) * f(x_up) < 0`.

    Args:
        f: Function whose zero we wish to determine.
        x_low: lower endpoint of the interval in which the zero is sought.
        x_up: upper endpoint of the interval in which the zero is sought.
        tol_abs: Absolute error target.
        tol_rel: Relative error target.
        max_iters: Maximum number of iterations.

    Returns:
        An approximation of a zero of :math:`f` and its estimated error.

    Examples:
        
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> x_low, x_up = 0.0, 1.0
        >>> f_zero, error = bisection(f, x_low, x_up)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '        
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = nan (nan),   f(f_zero) = nan.
        
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> x_low, x_up = 5.0, 6.0
        >>> f_zero, error = bisection(f, x_low, x_up)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = 5.267 (9.5e-07),   f(f_zero) = -6.31e-07.
    
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> x_low, x_up = -10.0, -9.0
        >>> f_zero, error = bisection(f, x_low, x_up)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = -9.679 (9.5e-07),   f(f_zero) = -1.64e-07.
     
    """


def taylor_approximation(
    x: np.ndarray, 
    function: Callable[[float], float], 
    function_derivative: Callable[[float, int], float], 
    K: int,
    x_0: float = 0.0,
) -> np.ndarray:
    """ Taylor approximation of a function

    Args:
        x: Point at which the approximation es evaluated. 
        function: Function to be approximated.  
        function_derivative: Derivative of the function.
        K: Order of the approximation (degree of the Taylor polynomial).
        x_0: Point about which the approximation is made.

    Returns:
        Array with the Taylor approximations of the function for different orders. 
        The kth row in the array yields the order k approximation of the function.

    Example:    
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> y = taylor_approximation(
        ...         x,    
        ...         function=lambda x: np.exp(- x), 
        ...         function_derivative=lambda x, k: (1- 2 * (k % 2)) * np.exp(- x), 
        ...         K=8,
        ...     )
        >>> print(np.round(y, 6))
        [[1.       1.       1.      ]
         [2.       1.       0.      ]
         [2.5      1.       0.5     ]
         [2.666667 1.       0.333333]
         [2.708333 1.       0.375   ]
         [2.716667 1.       0.366667]
         [2.718056 1.       0.368056]
         [2.718254 1.       0.367857]
         [2.718279 1.       0.367882]]
        >>> print(np.round(np.exp(- x), 6))
        [2.718282 1.       0.367879]
    """ 
    approximation = np.empty((K+1, len(x)))
    approximation[0, :] = function(x_0)
    
    x -= x_0 # Center at x_0
    
    for k in range(1, K+1):
        approximation[k, :] = (
            approximation[k-1, :] 
            + function_derivative(x_0, k) * x**k / math.factorial(k)
        )
        
    return approximation


if __name__ == "__main__":
    import doctest
    doctest.testmod()
