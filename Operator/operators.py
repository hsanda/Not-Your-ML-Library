import numpy as np

class Add():
    
    def add(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Subtract():

    def subtract(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Multiplication():
    
    def multiplication(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Divide():
    
    def divide(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Matrix_Multiplication():
    
    def mat_mul(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Sum():
    
    def sum(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Power():
    
    def pow(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Exponential():
    
    def exp(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Square_Root():
    
    def sqrt(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Absolute_Value():
    
    def abs(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Logirithm():
    
    def log(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Max():
    
    def max(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Min():
    
    def min(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Sine():
    
    def sin(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Cosine():
    
    def cos(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Tangent():
    
    def tan(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Hyperbolic_Sine():
    
    def sinh(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Hyperbolic_Cosine():
    
    def cosh(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Hyperbolic_Tangent(): 
    
    def tanh(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Mean():
    
    def mean(self):
        return jnp.sum
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Median():
    
    def median(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Mode():
    
    def mode(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Covariance():
    def __init__(self) -> None:
        pass
    
    def cov(self, x1, x2) -> np.ndarray:
        """
        inputs:
            x1: np.ndarray - vector that you wish to calculate the covariance of. shape: (n, d)
            x2: np.ndarray - vector that you wish to calculate the covariance relative to. shape: (n, d)
        
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
        mu_x1 = np.mean(x1)
        mu_x2 = np.mean(x2)
        
        return (1 / (N - 1)) * np.sum((x1 - mu_x1) * (x2 - mu_x2))
    
class Standard_Deviation():
    
    def std_dev(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Maxpool():
    
    def maxpool(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Cross_Correlation():
    
    def cross_correlation(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Concatenate():
    
    def concatenate(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Expand_Dimensions():
    
    def expand_dims(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Slice():
    
    def slice(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    