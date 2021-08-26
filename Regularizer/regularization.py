import jax.numpy as jnp

class Regularization():
    def __init__(self):
        pass
    
    def l1_norm(self, x):
        """
        inputs:          
            x -> jnp.array:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        return jnp.sum(jnp.abs(x))
    
    def l2_norm(self, x):
        """
        inputs:          
            x -> jnp.array:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        return jnp.sqrt(jnp.sum(jnp.square(x)))
    
    def p_norm(self, x, p):
        """
        inputs:          
            x -> jnp.array:
            p -> float:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        return jnp.sum(jnp.power(jnp.abs(x), p))
    
    def l1_regularizer(self, x, lambda_):
        """
        inputs:            
            x -> jnp.array: 
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            \lambda * \sum_{i=1}^{N} \left | x_{i} \right |
        """
        return lambda_ * self.l1_norm(x)
    
    def l2_regularizer(self, x, lambda_):
        """
        inputs:            
            x -> jnp.array: 
            lambda_ -> float:
            
        returns:

            
        short description:
        
        
        description:
            function in latex
            \lambda * \sum_{i=1}^{N} (x_{i})^2
        """
        return lambda_ * self.l2_norm(x)
    
    def lp_regularizer(self, x, lambda_):
        """
        inputs:          
            x -> jnp.array:  
            lambda_ -> float:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            \sum_{i=1}^{N} \left | x_{i} \right |^p
        """
        return lambda_ * jnp.power(self.p_norm(), 1/p)