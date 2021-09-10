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
    
    def calculate_std_dev(self):
        pass