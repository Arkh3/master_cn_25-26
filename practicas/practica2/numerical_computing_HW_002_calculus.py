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
import mpmath as mp
import matplotlib.pyplot as plt

from scipy.stats import norm
from typing import Callable, Tuple, Union


########################### 1 ###########################

# Plot of Taylor approximations of different orders
def plot_taylor_approximation(
    function: Callable[[float], float], 
    function_derivative: Callable[[float, int], float], 
    K: int,
    x_0: float,
    interval_plot: Tuple[float, float],
    figure_size: Tuple[int, int],
) -> None:
    
    x = np.linspace(*interval_plot, num=1000)
    y_exact = function(x)
    y_taylor = taylor_approximation(x, function, function_derivative, K, x_0)
    
    fig, axs = plt.subplots(K + 1, 2, sharex=True, figsize=figure_size)
    
    for k, y in enumerate(y_taylor):
        axs[k, 0].plot(x, y_exact, label='exact')
        axs[k, 0].plot(x, y_taylor[k, :], label='Taylor(k = {:d})'.format(k))
        axs[k, 0].set_xlabel('$x$')
        axs[k, 0].set_ylabel('$f(x)$')
        axs[k, 0].legend()
    
        error = y_taylor[k, :] - y_exact 
        axs[k, 1].plot(x, error, label='error')
        axs[k, 1].set_xlabel('$x$')
        axs[k, 1].set_ylabel('$error$')
        axs[k, 1].legend()
        axs[k, 1].axhline(y = 0.0, color = 'k', linestyle = ':')

        
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
    
    for k in range(1, K+1):
        approximation[k, :] = approximation[k-1, :] + \
            function_derivative(x_0, k) * (x - x_0)**k / math.factorial(k)
    
    return approximation


########################### 2 ###########################

def plot_taylor_approximation_with_series_term(
    function: Callable[[float], float], 
    series_term: Callable[[float, int], float], 
    K: int,
    x_0: float,
    interval_plot: Tuple[float, float],
    figure_size: Tuple[int, int],
) -> None:
    
    x = np.linspace(*interval_plot, num=1000)
    y_exact = function(x)
    y_taylor = taylor_approximation_with_series_term(x, function, series_term, K) # x_0 is always taken as 0.0
    
    fig, axs = plt.subplots(K + 1, 2, sharex=True, figsize=figure_size)
    
    for k, y in enumerate(y_taylor):
        axs[k, 0].plot(x, y_exact, label='exact')
        axs[k, 0].plot(x, y_taylor[k, :], label='Taylor(k = {:d})'.format(k))
        axs[k, 0].set_xlabel('$x$')
        axs[k, 0].set_ylabel('$f(x)$')
        axs[k, 0].legend()
    
        error = y_taylor[k, :] - y_exact 
        axs[k, 1].plot(x, error, label='error')
        axs[k, 1].set_xlabel('$x$')
        axs[k, 1].set_ylabel('$error$')
        axs[k, 1].legend()
        axs[k, 1].axhline(y = 0.0, color = 'k', linestyle = ':')

 
def taylor_approximation_with_series_term(
    x: np.ndarray, 
    function: Callable[[float], float], 
    series_term: Callable[[float, int], float], 
    K: int,
    x_0: float = 0.0
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
    approximation[0, :] = series_term(x, 0)
    
    for k in range(1, K + 1):
        approximation[k, :] = approximation[k-1, :] + series_term(x, k)
    
    return approximation


########################### 3 ###########################

def stirling_approx(n, order):
    """ Return log(n!) using Stirling's approximation up to a given order """
    
    # base of the approximation
    approx = n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
    
    # coefficients for 1/(12n), -1/(360n^3), 1/(1260n^5), -1/(1680n^7), ...
    bernoulli_coeffs = [1/12, -1/360, 1/1260, -1/1680, 1/1188]
    
    for k in range(order):
        term = bernoulli_coeffs[k] / (n ** (2*k + 1))
        approx += term
    
    return approx


def plot_stirling_errors(orders):
    # Compare errors
    N_values = np.arange(10, 500, 10)  # factorial sizes
    errors = {order: [] for order in orders}

    for n in N_values:
        log_fact_exact = math.log(math.factorial(n))
        for order in orders:
            log_fact_approx = stirling_approx(n, order)
            # absolute error in log domain
            err = abs(log_fact_exact - log_fact_approx)
            errors[order].append(err)

    # Plot
    plt.figure(figsize=(8,6))
    for order in orders:
        plt.plot(N_values, errors[order], label=f"Order {order}")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Absolute error in log(n!)")
    plt.title("Error of Stirling's approximation with different truncation orders")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()


def stirling_approx_order_zero(n):
    """Zeroth-order Stirling approximation of n!"""
    return math.sqrt(2 * mp.pi * n) * (n / mp.e) ** n


def find_stirlings_relative_error(relative_error_thresholds):
    MAX_ITER = 100000
    
    # thresholds
    results = {}

    # search for each threshold
    for thr in relative_error_thresholds:
        
        N = 1
        while N < MAX_ITER:
            # Compute relative error
            fact = math.factorial(N)
            approx = stirling_approx_order_zero(N)
            err = abs(fact - approx) / fact
            
            if err < thr:
                results[thr] = N
                break
                
            N += 1
        if N == MAX_ITER:
            print("WARN: the loop exited because it reached the maximun number of iterations.")

    return results



########################### 5 ###########################

def bisection(
    f: Callable,
    x_low: float,
    x_up: float,
    tol_abs: float = 1.0e-6,
    max_iters: int = 100
) -> float:
    r""" Zero of the function :math:`f` using the bisection method.

    Find a zero in the interval :math:`\left( x_{low}, x_{up} \right)` 
    assuming that :math:`f(x_low) * f(x_up) < 0`.

    Args:
        f: Function whose zero we wish to determine.
        x_low: lower endpoint of the interval in which the zero is sought.
        x_up: upper endpoint of the interval in which the zero is sought.
        tol_abs: Absolute error target.
        max_iters: Maximum number of iterations.

    Returns:
        An approximation of a zero of :math:`f` and its estimated error.

    Examples:
        
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> x_low, x_up = 0.0, 1.0
        >>> f_zero, error = bisection(f, x_low=0.0, x_up=1.0, tol_abs=1.0e-6)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '        
        >>> format_string += 'f(f_zero) = {:.2e}'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = nan (nan),   f(f_zero) = nan
        
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> x_low, x_up = 5.0, 6.0
        >>> f_zero, error = bisection(f, x_low, x_up, tol_abs=1.0e-6)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = 5.267 (9.5e-07),   f(f_zero) = -6.31e-07
    
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> x_low, x_up = -10.0, -9.0
        >>> f_zero, error = bisection(f, x_low, x_up, tol_abs=1.0e-6)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = -9.679 (9.5e-07),   f(f_zero) = -1.64e-07
     
    """
    
    if (f(x_low) * f(x_up)) > 0.0:
        warnings.warn('No sign change in [{}, {}]'.format(x_low, x_up))
        return np.nan, np.nan 
    
    n_iter = 0
    x_mid = 0.5 * (x_low + x_up)    
    delta_x = 0.5 * (x_up - x_low)
    
    #  Your code goes here
    
    while (delta_x > tol_abs) and n_iter < max_iters:
        x_mid = (x_up + x_low) / 2
        delta_x = (x_up - x_low) / 2

        if f(x_mid) * f(x_up) < 0:
            x_low = x_mid

        elif f(x_mid) * f(x_low) < 0:
            x_up = x_mid
            
        else:
            x_low = x_mid
            x_up = x_mid

        n_iter += 1
    
    #####

    if n_iter == max_iters: 
        warnings.warn('Maximum number of iterations reached.')
     
    return x_mid, delta_x




def newton_raphson(
    f: Callable,
    df_dx: Callable,
    seed: float,
    tol_abs: float = 1.0e-6,
    max_iters: int = 50,
) -> float:
    """ Zero of the function :math:`f` using the Newton-Raphson method.

    Args:
        f: Function whose zero we wish to determine.
        df_dx: Derivative of :math:`f`.
        seed: Initial estimate of the zero (close to an actual one).
        tol_abs: Absolute error target.
        max_iters: Maximum number of iterations.

    Returns:
        An approximation of a zero of :math:`f` and its estimated error.

    Examples:
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> df_dx = lambda x: -np.sin(x) - 0.1
        >>> seed = 0.1
        >>> f_zero, error = newton_raphson(f, df_dx, seed, tol_abs=1.0e-6)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = 5.267 (1.1e-09),   f(f_zero) = -2.22e-16
        
     
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> df_dx = lambda x: -np.sin(x) - 0.1
        >>> seed = np.pi
        >>> f_zero, error = newton_raphson(f, df_dx, seed, tol_abs=1.0e-6)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = -9.679 (2.8e-09),   f(f_zero) = 3.33e-16
    
        >>> f = lambda x: np.cos(x)
        >>> df_dx = lambda x: -np.sin(x)
        >>> seed = 0.0
        >>> f_zero, error = newton_raphson(f, df_dx, seed, tol_abs=1.0e-6)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = nan (nan),   f(f_zero) = nan
        
    """
    
    if tol_abs is None:
        tol_abs = np.finfo(float).tiny
        
    n_iter = 0
    x = seed
    
    if np.isclose(df_dx(x), 0.0):
        warnings.warn('Newton-Raphson does not converge with this seed')
        return np.nan, np.nan
    
    #  TO DO: Your code goes here
    
    delta_x = f(x) / df_dx(x)

    while (abs(delta_x) > tol_abs) and n_iter < max_iters:
        if df_dx(x) == 0:
            return float('NaN'), float('NaN')
            
        x = x - delta_x
        
        delta_x = f(x) / df_dx(x)
        
        n_iter += 1
    
    ####
    
    if n_iter == max_iters: 
        warnings.warn('Maximum number of iterations reached.')
     
    return x, np.abs(delta_x)


def secant(
    f: Callable,
    a: float,
    b: float,
    tol_abs: float = 1.0e-6,
    max_iters: int = 100
) -> float:
    
    if (f(a) * f(b)) > 0.0 or a > b:
        warnings.warn('No sign change in [{}, {}]'.format(a, b))
        return np.nan, np.nan 
    
    n_iter = 0
    
    while b - a > tol_abs and n_iter < max_iters:
        
        c = a - f(a) * (b - a) / (f(b) - f(a))
        
        a = min(b, c)
        b = max(b, c)

        n_iter += 1

    if n_iter == max_iters: 
        warnings.warn('Maximum number of iterations reached.')
     
    return c, b - a



def vectorized_newton_raphson(
    f: Callable[[np.ndarray], np.ndarray],
    df_dx: Callable[[np.ndarray], np.ndarray],
    seed: Union[float, np.ndarray],
    tol_abs: float = 1.0e-6,
    tol_rel: float = None,
    max_iters: int = 50
) -> np.ndarray:
    """
    Vectorized Newton-Raphson method.
    Works for a single float seed or a numpy array of seeds.
    Returns (roots, last_update).
    """

    # Ensure seed is an array
    x = np.atleast_1d(seed).astype(float)

    # Initialize update and iteration counter
    delta = np.full_like(x, np.inf)
    iters = 0

    # Mask of "still converging" points
    active = np.ones_like(x, dtype=bool)

    while np.any(active) and iters < max_iters:
        f_val = f(x)
        df_val = df_dx(x)
        
        # If derivative is zero → mark as NaN
        if np.any(df_val == 0):
            x[active][zero_deriv] = np.nan
            delta[active][zero_deriv] = np.nan
            active[np.where(active)[0][zero_deriv]] = False
            continue

        delta_new = f_val[active] / df_val[active]
        x[active] = x[active] - delta_new
        delta[active] = delta_new

        # Check convergence
        conv = np.abs(delta_new) <= tol_abs
        active[np.where(active)[0][conv]] = False

        iters += 1

    # If user passed a scalar, return scalars
    if np.isscalar(seed):
        return x.item(), delta.item()
    
    return x, delta


def generate_gaussian_sample_with_inverse_method(size=100):
    # Sample ~ U(0,1) 
    u = np.random.uniform(0, 1, size=size)

    # Define the functions for the Newton Raphson step
    f = lambda x: norm.cdf(x) - u
    df_dx = lambda x: norm.pdf(x)
    
    seed = np.zeros_like(u)# for a N(0,1) 0 is a good estimate of the actual root
    
    f_zero, error = vectorized_newton_raphson(f, df_dx, seed=seed) 

    return f_zero



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    if __name__ == "__main__":
        import doctest
        doctest.testmod()
