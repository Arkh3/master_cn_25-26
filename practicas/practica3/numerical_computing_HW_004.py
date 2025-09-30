# -*- coding: utf-8 -*-
"""
Utilities for the course on Numerical Computing 
(Master's program in Data Science)

Created on Sun 2024-09-22

@author: alberto.suarez@uam.es
"""


import numpy as np
import warnings

from typing import Callable, Literal
from numpy.typing import ArrayLike, NDArray


def multivariate_normal_pdf(
    X: ArrayLike,
    mu: NDArray,
    Sigma: NDArray,
) -> ArrayLike:
    """ Multivariate normal probability density function


        :math:`normpdf\left(x; \boldsymbol{\mu}, \boldsymbol{\Sigma} \right) 
            = \exp\left(- (\mathbf{x} - \mathbf{mu} \right)^{\top} 
                  \boldsymbol{\Sigma} (\mathbf{x} - \mathbf{mu} \right) \right)
               \ \left(2 \pi \left| \boldsymbol{\Sigma} \right| \right)^{D/2} 
        
    
    Args:
        X:  Data matrix (array of dimension :math:`N \times D`)
        mu: Mean vector  (array of dimension :math:`D`)
        Sigma: Covariance matrix (array of dimension :math:`D \times D`)
        
    Returns:
        Value of the multivariate normal pdf (array of dimension :math:`N`) .


    Examples:
        
        # Example 1
        >>> from scipy.stats import multivariate_normal
        >>> mu = [-1.0, 5.0]
        >>> Sigma = [[3.0, 2.0],
        ...          [2.0, 5.0]]
        >>> X = [-2.0, 4.3]
        >>> multivariate_normal.pdf(X, mu, Sigma)
        0.040613982113634844
        >>> multivariate_normal_pdf(X, mu, Sigma)
        0.040613982113634844
        
        # Example 2
        >>> X = [[-1.5, 2.3],
        ...      [ 2.0, 0.0]]
        >>> multivariate_normal.pdf(X, mu, Sigma)
        array([2.14446679e-02, 1.34216312e-05])
        >>> multivariate_normal_pdf(X, mu, Sigma)
        array([2.14446679e-02, 1.34216312e-05])


        # Example 3
        >>> X = [[-1.5, 2.3],
        ...      [ 2.0, 0.0],
        ...      [-5.0, 1.0]]
        >>> multivariate_normal.pdf(X, mu, Sigma)
        array([2.14446679e-02, 1.34216312e-05, 2.61650555e-03])
        >>> multivariate_normal_pdf(X, mu, Sigma)
        array([2.14446679e-02, 1.34216312e-05, 2.61650555e-03])
        
    """
    D = np.shape(X)[-1]
    X = np.asarray(X) - np.asarray(mu)
    return (
        np.exp(-0.5 * np.sum(X.T * np.linalg.solve(Sigma, X.T), axis=0))
        / np.sqrt((2.0 * np.pi)**D  * np.linalg.det(Sigma)) 
    )
    
def rotation_matrix_2D(theta: float) -> NDArray:
    """ Rotation matrix in 2 dimensions (counterclockwise)

        
        rotation_matrix_2D(theta) = [[cos(theta), -sin(theta)],
                                     [sin(theta), cos(theta)]]        
    
    Args:
        theta: Covariance matrix (array of dimension :math:`D \times D`)
        
    Returns:
        Value of the multivariate normal pdf (array of dimension :math:`N`) .

    Examples:
        
        # Example 1: 
        >>> rotation_matrix_2D(theta=0.0)
        array([[ 1., -0.],
               [ 0.,  1.]])

        # Example 2: 
        >>> rotation_matrix_2D(theta=np.pi / 2.0) @ [1.0, 0.0]
        array([6.123234e-17, 1.000000e+00])
        
        # Example 3: 
        >>> rotation_matrix_2D(theta=np.pi / 4.0)  @ [1.0, 0.0]
        array([0.70710678, 0.70710678])
        
        # Example 4: 
        >>> rotation_matrix_2D(theta=np.pi / 3.0)  
        array([[ 0.5      , -0.8660254],
               [ 0.8660254,  0.5      ]])
        
        
    """    
    return np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )

def remove_sign_column_vectors(matrix: NDArray) -> NDArray:
    """ Makes positive the 1st non-zero element of each column of a matrix

        
    Args:
        matrix: A general real matrix      
    Returns:
        matrix with the signs of the corresponding the columns changed if 
        the first non-zero element was negative.

    Examples:
        
        # Example 1: 
        >>> remove_sign_column_vectors(
        ...     np.array(
        ...         [[1.0, -1.0,   0.0,  0.0,  0.0],
        ...          [1.0, -1.0,  -1.0,  0.0,  0.0],
        ...          [1.0, -1.0,  -1.0,  1.0, -1.0],
        ...          [1.0, -1.0,  -1.0,  1.0, -1.0]]
        ...     )
        ... )
        array([[1., 1., 0., 0., 0.],
               [1., 1., 1., 0., 0.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]])
        
    """
    
    n_rows, n_columns = np.shape(matrix)
    for index_column in range(n_columns):
        index_row = 0
        while (
                (index_row < n_rows) 
                and 
                (matrix[index_row, index_column] == 0.0)
            ):
            index_row += 1
    
        matrix[index_row:, index_column] *= (
            np.sign(matrix[index_row, index_column])
        )
    return matrix



def make_data_matrix(
        X: NDArray, 
        include_bias: bool = False
    ) -> NDArray:
    """ Prepare data for fit
   
        Buils a data matrix of size (n_samples, dimension)
        if `include_bias` is `False` 

        Buils a data matrix with of size (n_samples, dimension + 1) 
        if `include_bias` is `True` 

    Args:
        X: Either data vector of size (n_samples)  
           or a data matrix of size (n_samples, dimension)
           
        include_bias: Whether to calculate the intercept for this model. 
                      If set to False, no intercept is used in calculations 
                      (i.e., the data is expected to be centered).
       
       
    Returns:
        (n_samples, dimension, Data matrix)
        The data matrix is
           of size (n_samples, dimension) if `include_bias` is `False` 
           of size (n_samples, dimension + 1) if `include_bias` is `True`
   
    Note:
        Should return the same value as the `fit` method of 
        the class `sklearn.linear_model.LinearRegression`
        https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LinearRegression.html
       
    Examples:
       
        # Example 1: 
        >>> make_data_matrix([1.0, 2.0, 3.0])           # no bias term
        (3, 1, array([[1.],
               [2.],
               [3.]]))
        
        # Example 2: 
        >>> make_data_matrix([1.0, 2.0, 3.0], include_bias=True)
        (3, 2, array([[1., 1.],
               [1., 2.],
               [1., 3.]]))
      
        # Example 3: 
        >>> X = np.reshape(np.arange(15), (5, 3))
        >>> make_data_matrix(X)
        (5, 3, array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11],
               [12, 13, 14]]))

        # Example 4: 
        >>> X = np.reshape(np.arange(15), (5, 3))
        >>> make_data_matrix(X, include_bias=True)
        (5, 4, array([[ 1.,  0.,  1.,  2.],
               [ 1.,  3.,  4.,  5.],
               [ 1.,  6.,  7.,  8.],
               [ 1.,  9., 10., 11.],
               [ 1., 12., 13., 14.]]))
      
    """
    
    data_matrix = np.asarray(X)
    
    if data_matrix.ndim == 1:
        data_matrix = np.atleast_2d(data_matrix).T
        
    n_samples, dimension = np.shape(data_matrix)
   
    if include_bias:
        dimension +=1
        data_matrix = np.column_stack(
            (np.ones(n_samples), data_matrix)
        )
    
    return n_samples, dimension, data_matrix


def linear_fit(
        X: NDArray, 
        y: NDArray, 
        regularization_rate: int | float = 0, 
        fit_intercept: bool = True
    ) -> NDArray:
    """ Linear fit by least squares with L2 regularization
    
        Algebraic solution of a linear fit using regularized least-squares
        
    Args:
        X: Data matrix of size (n_samples, dimension)
        y: Vector of outputs of size (n_samples)
        regularization_rate: strength of the regularization. 
                             The default value is 0 (no regularization). 
        fit_intercept: Whether to calculate the intercept for this model. 
        If set to False, no intercept will be used in calculations 
        (i.e. data is expected to be centered).
        
        
    Returns:
        Vector of linear regression coefficients
    
    Note:
        Should return the same value as the `fit` method of 
        the class `sklearn.linear_model.LinearRegression`
        https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LinearRegression.html
        
    Examples:
        
        # Example 1: 
        >>> X = np.array([1.0, 2.0, 3.0])           # 1-D data
        >>> regression_coefficient = -3.0
        >>> y = regression_coefficient * X          # No intercept
        >>> linear_fit(X, y, fit_intercept=False)
        array([-3.])
       
        # Example 2: 
        >>> X = np.array([1.0, 2.0, 3.0])           # 1-D data
        >>> regression_coefficients = [7.0, -3.0]
        >>> y = regression_coefficients[0] - regression_coefficients[1] * X
        >>> linear_fit(X, y)
        array([7., 3.])
        
        # Example 3: 
        >>> rng = np.random.default_rng(seed=42)
        >>> X = rng.random((10, 2))                 # 2-D data
        >>> regression_coefficients = [3.0, -5.0]
        >>> y =  X @ regression_coefficients       # No intercept 
        >>> linear_fit(X, y, fit_intercept=False)
        array([ 3., -5.])
        
        # Example 4: 
        >>> rng = np.random.default_rng(seed=42)    # 2-D data
        >>> X = rng.random((10, 2))
        >>> regression_coefficients = [3.0, -5.0]
        >>> y = 4.0 + X @ regression_coefficients
        >>> linear_fit(X, y, fit_intercept=True)
        array([ 4.,  3., -5.])
        
        # Example 5: 
        >>> from sklearn.linear_model import LinearRegression
        >>> rng = np.random.default_rng(seed=42)
        >>> n_samples, dimension = (10, 2)
        >>> X = rng.random((n_samples, dimension))  # 2-D noisy data
        >>> y = 4.0 - X @ [-3.0, 5.0] + rng.standard_normal(n_samples)
        
        >>> reg = LinearRegression().fit(X, y)
        >>> print(reg.intercept_, reg.coef_)
        4.41438477375772 [ 2.35577944 -4.91126014]
        
        >>> linear_fit(X, y, fit_intercept=True)
        array([ 4.41438477,  2.35577944, -4.91126014])
 
        
    """
    _, dimension, X =  make_data_matrix(X, fit_intercept)
    
    cov_matrix_X_X = X.T @ X
    
    if regularization_rate:
        cov_matrix_X_X += regularization_rate * np.eye(dimension)
    
    cov_vector_X_y = X.T @ y
    return np.linalg.solve(cov_matrix_X_X, cov_vector_X_y)


def polynomial_features(
        X: NDArray, 
        degree: int, 
        include_bias: bool
    ) -> NDArray:
    """ Polynomial feature embedding
        
    Args:
        X: Data vector of size (n_samples)
        degree: degree of the polynomial
        include_bias: whether to include a bias term in the embedding
        
    Returns:
        Data matrix in the feature space
            If include_bias is False, its size is: (n_samples, degree)
            If include_bias is True, its size is: (n_samples, degree + 1)

    Note:      
        The present implementation works only for 1-D data.
        
    Examples:
        
        # Example 1: 
        >>> polynomial_features([1, 2, 3, 4], degree=3, include_bias=False)
        array([[ 1.,  1.,  1.],
               [ 2.,  4.,  8.],
               [ 3.,  9., 27.],
               [ 4., 16., 64.]])
        
        # Example 2: 
        >>> polynomial_features([1, 2, 3, 4], degree=3, include_bias=True)
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  2.,  4.,  8.],
               [ 1.,  3.,  9., 27.],
               [ 1.,  4., 16., 64.]])
 
        # Example 3: 
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> polynomial_features(X, degree=2, include_bias=False)
        array([[1, 2],
               [3, 4],
               [5, 6]])
        
        TODO: Implement polynomial features for multivariate data.
              The result of this evaluation should be:
                  array([[1., 2.,  1.,  2.,  4.],
                         [3., 4.,  9., 12., 16.],
                         [5., 6., 25., 30., 36.]])
           
        # Example 4: 
        >>> X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        >>> polynomial_features(X, degree=2, include_bias=True)
        array([[1., 1., 2.],
               [1., 3., 4.],
               [1., 5., 6.]])
        
        TODO: Implement polynomial features for multivariate data        
              The result of this evaluation should be:
                  array([[1., 1., 2.,  1.,  2.,  4.],
                         [1., 3., 4.,  9., 12., 16.],
                         [1., 5., 6., 25., 30., 36.]])
           
              
    """
    X = np.asarray(X)
    if X.ndim == 1:
        n_samples = len(X)
        if include_bias:
            dimension = degree + 1
            X_embedded = np.ones((n_samples, dimension))   
        else: 
            dimension = degree
            X_embedded = np.empty((n_samples, dimension))
            X_embedded[:, 0] = X
    
        for d in np.arange(1, dimension):
            X_embedded[:, d] = X_embedded[:, d - 1] * X
    else:
        n_samples, dimension = np.shape(X)
        
        if include_bias:
            X_embedded = np.column_stack((np.ones(n_samples), X))
        else:
            X_embedded = X
            
        warnings.warn('Only data in 1-D can be embedded at the moment')
    
    return X_embedded


def random_Fourier_features(
    X: NDArray, 
    n_features: int, 
    generate_random_sample: Callable,
    scale: float = 1.0, 
    include_bias: bool = False, 
    type: Literal['sin-cos', 'dephased cos'] =  'sin-cos'
) -> NDArray:
    """ Random Fourier feature embedding
    
        Computes random Fourier features 
        either sine-cosine, or dephased-cosine   
         
    Args:
        X: Data matrix of size (n_samples, dimension)
        n_features: number of features to be built
        scale: scale of the random frecuencies
        include_bias: whether to include a bias term in the embedding
        generate_random_sample: Random number generator 
        
    Returns:
        data matrix in the feature space
            If include_bias is False, its size is: (n_samples, n_features)
            If include_bias is True, its size is:  (n_samples, n_features + 1)

    References:
        
        Ali Rahimi, Benjamin Recht:
        Random Features for Large-Scale Kernel Machines. 
        NIPS 2007: 1177-1184
        https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/ 
        
    Examples:
        
        # Example 1: 
        >>> X = [1.0, 2.0, 3.0, 4.0]    
        >>> random_Fourier_features(
        ...    X, 
        ...    2, 
        ...    np.random.default_rng(seed=123).standard_normal,
        ...    include_bias=True, 
        ... )
        array([[1.        , 0.06829936, 0.99766487],
               [1.        , 0.13627974, 0.9906704 ],
               [1.        , 0.20362367, 0.97904923],
               [1.        , 0.27001661, 0.96285566]])

        >>> X = [[1, 2], [3, 4], [5, 6]] 
        >>> generate_random_sample = (
        ...    np.random.default_rng(seed=123).standard_normal
        ... )
        >>> random_Fourier_features(X, 4, generate_random_sample)
        array([[-0.5183629 ,  0.12634418, -0.85516075,  0.99198647],
               [ 0.91609223, -0.88387651,  0.40096761, -0.46772034],
               [-0.97992424,  0.89125383,  0.19937022, -0.45350482]])
        

    """

    X = np.asarray(X)
    
    if X.ndim == 1:
        X_aux = np.atleast_2d(X).T
    else:
        X_aux = X
        
    n_samples, dimension = np.shape(X_aux)
    
    if type == 'sin-cos':
        W = 2.0 * np. pi * scale * generate_random_sample(
            size=(dimension, int(np.ceil(n_features // 2)))
        )
        angles = X_aux @ W
        if include_bias:
            X_embedded =  np.column_stack(
                (np.ones(n_samples), 
                 np.sin(angles), 
                 np.cos(angles))
                )
        else:
            return np.column_stack((np.sin(angles), np.cos(angles)))
    elif type == 'dephased-cos':
        warnings.warn('Type of random feature not supported yet')
        X_embedded = X
    else:
        warnings.warn('Type of random feature not supported yet')
        X_embedded = X
        
        
    return X_embedded
        


if __name__ == "__main__":
    import doctest
    doctest.testmod()
