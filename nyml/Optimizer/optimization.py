from typing import Callable
import numpy as np

class optimization():
    """
    class that contains all optimization methods. These methods are the following:
        First Order Algorithms:
            - Gradient Descent
                - Batch Gradient Descent
                - Stochastic Gradient Descent 
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
    
    
    def batch_iterator(self, data, size_of_batch):
            p = np.random.permutation(data.shape[0])
            data_rand = data[p]
            for i in np.arange(0, data.shape[0], size_of_batch):
                yield data_rand[i:i + size_of_batch]
    
    """
    By taking the gradients of all the points within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """    
    def gradient_descent(self, params:dict, lr:float, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        size_of_batch = len(data) # batch size is equal to the size of the dataset
        
        for i in range(epochs):
            for scrambled_dataset in self.batch_iterator(data, size_of_batch):
                d_param = self.eval_grads(loss_fun, params, scrambled_dataset)
                for param in params:
                    params[param] = params[param] - (lr * d_param[param]) # e.g. w = w - lr * d_w
                
        return params

    
    """
    By taking the gradient (only one gradient) at the current point within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def stochastic_gradient_descent(self, params:dict, lr:float, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        size_of_batch = 1
        
        for i in range(epochs):
            for random_data_point in self.batch_iterator(data, size_of_batch):
                d_param = self.eval_grads(loss_fun, params, random_data_point)
                for param in params:
                    params[param] = params[param] - (lr * d_param[param]) # e.g. w = w - lr * d_w
                
        return params
    
    """
    By taking the gradient of all the points within a (batch) portion within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        size_of_batch (int): size of the batch
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def mini_batch_gradient_descent(self, params:dict, lr:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_param = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    params[param] = params[param] - (lr * d_param[param]) # e.g. w = w - lr * d_w
                
        return params
    
    # ----------------------------------------------------------------------------------------------------------------------  
    """
    By taking the gradient of all the points within a (batch) portion within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        size_of_batch (int): size of the batch
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def momentum(self, params:dict, lr:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        v_k = 0
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_param = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    params[param] = params[param] + v_k # e.g. w = w + ((-1 * (lr * d_w)) + (gamma * v_k))
                    vk_1 = (-1 * (lr * d_param[param])) + (gamma * vk_1) # (gamma * vk_1) is the momentum
                    v_k = vk_1 # everything before was done for readability of the math. This line is to update the momentum var but isnt true to form for the math.  
                
        return params
    
    def nesterov_gradient_acceleration(self, params:dict, lr:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        v_k = 0
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_param = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    params[param] = params[param] + v_k # e.g. w = w + ((-1 * (lr * d_w)) + (gamma * v_k))
                    vk_1 = (-1 * (lr * d_param[param])) + (gamma * vk_1) # (gamma * vk_1) is the momentum
                    v_k = vk_1 # everything before was done for readability of the math. This line is to update the momentum var but isnt true to form for the math.  
                
        return params
    
    def adagrad(self):
        #TODO: implement adagrad
        pass
    
    # ----------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- Overloaded functions ---------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    
    """
    By taking the gradient (only one gradient) at the current point within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def SGD(self, params:dict, lr:float, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        size_of_batch = 1
        
        for i in range(epochs):
            for random_data_point in self.batch_iterator(data, size_of_batch):
                d_param = self.eval_grads(loss_fun, params, random_data_point)
                for param in params:
                    params[param] = params[param] - (lr * d_param[param]) # e.g. w = w - lr * d_w
                
        return params
    
    """
    By taking the gradient of all the points within a (batch) portion within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        size_of_batch (int): size of the batch
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def mini_batch_SGD(self, params:dict, lr:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_param = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    params[param] = params[param] - (lr * d_param[param]) # e.g. w = w - lr * d_w
                
        return params
    
    def NAG(self, params:dict, lr:float, past_lr:float, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        #TODO: implement NAG
        pass
    
    
    
    
    