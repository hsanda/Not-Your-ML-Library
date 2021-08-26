import jax.numpy as jnp
from regularization import l2_norm, euclidean_norm


class Loss_Functions():
    def __init__(self):
        pass
    
    def mean_square_error(self, y_true, y_pred):
        """
        inputs:
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            mse: mean square error
            
        short description:
            null
        
        description:
        function in latex
        MSE = \frac{1}{N}\sum_{i=1}^{N} (\hat{y_{i}} - y_{i})^2
        
        """
        
        return jnp.mean(jnp.square(y_pred - y_true), axis=-1)
    
    def root_mean_square_error(self, y_true, y_pred):
        """
        inputs:
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            rsme: root mean square error
            
        short description:
            null
        
        description:
            function in latex
            RMSE = \sqrt{ \frac{1}{N}\sum_{i=1}^{N} (\hat{y_{i}} - y_{i})^2}
        """
        
        return jnp.sqrt(self.mean_square_error(y_true, y_pred))
    
    def mean_absolute_error(self, y_true, y_pred):
        """
        inputs:            
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            mae: mean absolute error
            
        short description:
            null
        
        description:
            function in latex
            MAE = \frac{1}{N}\sum_{i=1}^{N} \left |(\hat{y_{i}} - y_{i})\right |
        """
        
        return jnp.mean(jnp.abs(y_pred - y_true), axis=-1)
    
    def mean_absolute_percentage_error(self, y_true, y_pred):
        """
        inputs:            
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            MAPE: mean absolute percentage error
            
        short description:
            null
        
        description:
            function in latex
            MAPE = \frac{1}{N}\sum_{i=1}^{N} \left |\frac{\hat{y_{i}} - y_{i}}{y_{i}}\right |
        """
        
        return jnp.mean(jnp.abs((y_true - y_pred) / y_true), axis=-1)
    
    def mean_squared_logarithmic_error(self, y_true, y_pred):
        """
        inputs:            
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            msle: mean squared logarithmic error
            
        short description:
            null
        
        description:
            function in latex
            MSLE = \frac{1}{N}\sum_{i=1}^{N} (\log (y_{i} + 1) - log(\hat{y_{i}} + 1))^{2}
        """
        
        return jnp.mean(jnp.square(jnp.log(y_pred + 1) - jnp.log(y_true + 1)), axis=-1)
    
    def mean_bias_error(self, y_true, y_pred):
        """
        inputs:            
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            MBE: mean bias error
            
        short description:
            null
        
        description:
            function in latex
            MBE = \frac{1}{N}\sum_{i=1}^{N} (y_{i} - \hat{y_{i}})
        """
        
        return jnp.mean((y_pred - y_true), axis=-1)
    
    def squared_error_loss(self, y_true, y_pred):
        """
        inputs:
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            se: squared error
            
        short description:
            null
        
        description:
        function in latex
        SE = \sum_{i=1}^{N} (\hat{y_{i}} - y_{i})^2
        
        """
        
        return jnp.square(y_pred - y_true)
    
    def cosine_similarity_loss(self, y_true, y_pred):
        """
        inputs:            
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            cos_sim: cosine similarity
            
        short description:
            null
        
        description:
            function in latex
            cos \ sim = \frac{\sum^N_{i=1} \hat{y_{i}} y_{i}}{\sqrt{\sum^N_{i=1} \hat{y_{i}}^2} \sqrt{\sum^N_{i=1} y_{i}^2}}
        """
        l2_norm_y_true = l2_norm(y_true)
        l2_norm_y_pred = l2_norm(y_pred)
        
        numerator = jnp.sum(y_true * y_pred)
        denominator = l2_norm_y_true * l2_norm_y_pred
        
        return numerator / denominator
    
    def logarithmic_loss(self, y_true, y_pred):
        """
        inputs:            
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            log_loss: logarithmic loss
            
        short description:
            null
        
        description:
            function in latex
            log \ loss = \sum^N_{i=1} \sum^M_{j=1} y_{ij} \log \hat{y}_{ij}
        """
        
        return jnp.sum(jnp.square(jnp.log(y_pred + 1) - jnp.log(y_true + 1)), axis=-1)
    
    def binary_logarithmic_loss(self, y_true, y_pred):
        """
        inputs:            
            y_true: true labels
            y_pred: predicted labels
        
        returns:
            b_log_loss: binary logarithmic loss
            
        short description:
            null
        
        
        description:
            function in latex
            bi \ log \ loss = -\frac{1}{N}\sum^N_{i=1} y_{i} \log \hat{y_{i}} + (1 - y_{i}) \log(1 - \hat{y_{i}})
        """
        
        return jnp.sum(jnp.square(jnp.log(y_pred + 1) - jnp.log(y_true + 1)), axis=-1)
    
    def logarithmic_hyperbolic_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.sum(jnp.square(jnp.log(jnp.abs(y_pred + 1) + 1) - jnp.log(jnp.abs(y_true + 1) + 1)), axis=-1)
    
    def huber_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.sum(jnp.square(jnp.sqrt(jnp.abs(y_pred - y_true)) - 1), axis=-1)
    
    def poisson_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.sum(jnp.square(y_pred - y_true), axis=-1)
    
    def categorical_cross_entropy(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(-y_true * jnp.log(y_pred), axis=-1)
    
    def hinge_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.maximum(1. - y_true * y_pred, 0.), axis=-1)
    
    def squared_hinge_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.maximum(jnp.square(1. - y_true * y_pred), 0.), axis=-1)
    
    def categorical_hinge_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.maximum(jnp.sum(y_true * y_pred, axis=-1) - 1., 0.), axis=-1)
    
    def binary_cross_entropy(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(-y_true * jnp.log(y_pred), axis=-1)
    
    def binary_logistic_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.log(jnp.maximum(1., y_pred)) - jnp.log(jnp.maximum(1., 1. - y_pred)), axis=-1)
    
    def sigmoid_binary_cross_entropy(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.maximum(jnp.sum(y_true * y_pred, axis=-1) - 1., 0.), axis=-1)
    
    def softmax_cross_entropy(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(-jnp.sum(y_true * jnp.log(y_pred), axis=-1))
    
    def multi_class_cross_entropy(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(-jnp.sum(y_true * jnp.log(y_pred), axis=-1))
    
    def sparse_multi_class_cross_entropy(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(-jnp.sum(y_true * jnp.log(y_pred), axis=-1))
    
    def multi_class_logistic_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.log(jnp.maximum(1., y_pred)) - jnp.log(jnp.maximum(1., 1. - y_pred)))
    
    def multi_class_sparsemax_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.maximum(jnp.sum(y_true * y_pred, axis=-1) - 1., 0.))
    
    def negative_log_likelihood(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))
    
    def kullback_leibler_divergence_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))
    
    def support_vector_machine_loss(self, y_true, y_pred):
        """
        inputs:            
        
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        
        return jnp.mean(jnp.maximum(1. - y_true * y_pred, 0.))
    
    