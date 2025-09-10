# -*- coding: utf-8 -*-
"""
Utilities for the course on Numerical Computing 
(Master's program in Data Science)

Created on Sun 2024-09-22

@author: alberto.suarez@uam.es
"""


import numpy as np

from typing import Callable
from numpy.typing import ArrayLike

class ChebyshevApproximation:
    """
    Chebyshev approximation to a function.
     
    Given a function func, lower and upper limits of the interval [a,b],
    and n_func_evals b, this class computes a Chebyshev approximation
    of the function.
    The method eval(x) yields the approximated function value.
    
        
    Attributes:
        a, b: Limits of the approximation interval [a, b].
        n_func_evals: Number of functional evaluations.
        func: Function to be approximated.

        
    Remarks:
        Adapted from: https://www.excamera.com/sphinx/article-chebyshev.html
               (retrieved 2024-10-02)
       
        
    Example: 
        
    >>> ch = ChebyshevApproximation(0, np.pi / 12, 7, np.sin)
    >>> ch.eval(0.1) 
    0.09983341664682888
    >>> ch.eval([0.1, 0.2]) # np.sin(0.1) = 0.09983341664682815
    array([0.09983342, 0.19866933])
    
    """
    def __init__(self, a: float, b: float, degree: int, func: Callable):
        self.a = a
        self.b = b
        self.func = func

        n_func_evals = degree + 1
        radius_interval = 0.5 * (b - a)
        center_interval = 0.5 * (b + a)
        f = [
            func(np.cos(np.pi * (k + 0.5) / n_func_evals) 
                 * radius_interval + center_interval ) 
            for k in range(n_func_evals)
        ]
        fac = 2.0 / n_func_evals
        self.c = [fac * sum([f[k] * np.cos(np.pi * j * (k + 0.5) / n_func_evals)
                  for k in range(n_func_evals)]) for j in range(n_func_evals)]

    def eval(self, x: ArrayLike) -> ArrayLike:
        a, b = self.a, self.b
        x = np.asarray(x)
        y = (2.0 * x - a - b) / (b - a)
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)      # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:     # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]   # Last step is different
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
