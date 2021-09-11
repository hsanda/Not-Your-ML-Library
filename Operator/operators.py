import jax.numpy as jnp

class Operators():
    def __init__(self) -> None:
        pass
    
    def covariance(self, x1, x2) -> jnp.ndarray:
        """
        inputs:
            x1: jnp.ndarray - vector that you wish to calculate the covariance of. shape: (n, d)
            x2: jnp.ndarray - vector that you wish to calculate the covariance relative to. shape: (n, d)
        
        returns:
            jnp.ndarray - covariance of x1 and x2
            
        short description:
            mu_x1: float - mean of x1
            mu_x2: float - mean of x2
            x1_i: float - components within vector x1
            x2_i: float - components within vector x2
            
            The covariance is determining the level to which two variables are correlated.
        
        description:
        function in latex
        \text{Cov}(x_{1},x_{2}) = {1 \over N-1}\sum_{i=1}^N (x_{1_{i}}-\mu_{x_{1}})(x_{{2}_i}-\mu_{x_{2}})
        
        """
        N = x1.shape[0]
        mu_x1 = jnp.mean(x1)
        mu_x2 = jnp.mean(x2)
        
        return (1 / (N - 1)) * jnp.sum((x1 - mu_x1) * (x2 - mu_x2))
    
    
    def covariance(self, x1, x2) -> jnp.ndarray:
        """
        inputs:
            x1: jnp.ndarray - vector that you wish to calculate the covariance of. shape: (n, d)
            x2: jnp.ndarray - vector that you wish to calculate the covariance relative to. shape: (n, d)
        
        returns:
            jnp.ndarray - covariance of x1 and x2
            
        short description:
            mu_x1: float - mean of x1
            mu_x2: float - mean of x2
            x1_i: float - components within vector x1
            x2_i: float - components within vector x2
            
            The covariance is determining the level to which two variables are correlated.
        
        description:
        function in latex
        \text{Cov}(x_{1},x_{2}) = {1 \over N-1}\sum_{i=1}^N (x_{1_{i}}-\mu_{x_{1}})(x_{{2}_i}-\mu_{x_{2}})
        
        """
        N = x1.shape[0]
        mu_x1 = jnp.mean(x1)
        mu_x2 = jnp.mean(x2)
        
        return (1 / (N - 1)) * jnp.sum((x1 - mu_x1) * (x2 - mu_x2))
    
    def std_dev(self):
        pass
    
    def matrix_multiplication(self):
        pass
    
    def cross_correlation(self):
        pass
    
    def multiplication(self):
        pass
    
    def divide(self):
        pass
    
    def max(self):
        pass
    
    def min(self):
        pass
    
    def maxpool(self):
        pass
    
    def sum(self):
        pass
    
    def add(self):
        pass
    
    def subtract(self):
        pass
    
    def pow(self):
        pass
    
    def exponential(self):
        pass
    
    def log(self):
        pass
    
    def sqrt(self):
        pass
    
    def sin(self):
        pass
    
    def cos(self):
        pass
    
    def tan(self):
        pass
    
    def sinh(self):
        pass
    
    def cosh(self):
        pass
    
    def tanh(self):
        pass
    
    def mean(self):
        return jnp.sum
    
    def median(self):
        pass
    
    def mode(self):
        pass
    
    def abs(self):
        pass
    
    def concatenate(self):
        pass
    
    def expand_dims(self):
        pass
    
    def slice(self):
        pass
    
    