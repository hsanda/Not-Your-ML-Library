import numpy as np

class optimization():
    """
    class that contains all optimization methods. These methods are the following:
        First Order Algorithms:
            - Gradient Descent
                - Stochastic Gradient Descent 
                - Batch Gradient Descent
                - Mini-Batch Gradient Descent
                - AdaGrad(Adaptive Gradient Descent)
                    - AdaDelta
                - Nesterov's Gradient acceleration
            - Conjugate Gradient
                - Conjugate Gradient
                - Conjugate Gradient with Restarts
            - Momentum
            - RMS-Prop (Root Mean Square Propagation)
            - Adam(Adaptive Moment Estimation)
                - Adam 
                - Adam with restarts
        
        Second Order Algorithms:
            - Newton's Method
            - Secant Method
            - Quasi-Newton Method
                - Davidson-Fletcher-Powell
                - Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                - Limited-memory BFGS (L-BFGS)
                - Newton-Raphson
                - Levenberg-Marquardt
                - Powell's method
                - Steepest Descent
                - Truncated Newton
                - Fletcher-Reeves
    """ 
    def gradient_descent(self, y_pred, y_true, lr, epochs, loss_function):
        
        for i in range(epochs):
            
    
    def stochastic_gradient_descent(self):
        pass
    
    def batch_gradient_descent(self):
        pass
    
    def mini_batch_gradient_descent(self):
        pass
    
    def adagrad(self):
        pass
    
    def nesterov_gradient_acceleration(self):
        pass
    
    
    
    
    