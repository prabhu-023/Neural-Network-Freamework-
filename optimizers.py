import numpy as np

class Optimizer_SGD_momentum():
    def __init__(self,learning_rate=1.0,decay=0.,momentum=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0
        
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = \
                self.learning_rate*(1./(1.+self.decay*self.iteration))
    def update_parm(self,layer):
        if self.momentum:
           if not hasattr(layer,'weight_momentum'):
               layer.weight_momentum = np.zeros_like(layer.weights)
               layer.biases_momentum = np.zeros_like(layer.biases)
           weight_updates = self.momentum*layer.weight_momentum -\
                (self.current_learning_rate*layer.dweights)
           layer.weight_momentum = weight_updates
           biase_updates = self.momentum*layer.biases_momentum -\
                (self.current_learning_rate*layer.dbiases)
           layer.biases_momentum = biase_updates
           
        else:
            weight_updates = -self.current_learning_rate * \
            layer.dweights
            biase_updates = -self.current_learning_rate * \
            layer.dbiases
            
        layer.weights +=weight_updates
        layer.biases +=biase_updates
    def post_update(self):
        self.iteration+=1
class Optimizer_Ada_Grad():
    def __init__(self,learning_rate=1.0,decay=0.,epsilon=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iteration = 0
        
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = \
                self.learning_rate*(1./(1.+ self.decay*self.iteration))
    def update_parm(self,layer):
        if self.epsilon:
            if not hasattr(layer,'weight_momentum'):
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.biases_cache = np.zeros_like(layer.biases)
            layer.weight_cache +=  layer.dweights**2
            layer.biases_cache += layer.dbiases**2
            layer.weights +=-self.current_learning_rate*layer.dweights\
                /(np.sqrt(layer.weight_cache)+self.epsilon)
            layer.biases +=-self.current_learning_rate*layer.dbiases\
                /(np.sqrt(layer.biases_cache)+self.epsilon)
        
    def post_update(self):
        self.iteration+=1

class Optimizer_RMSprop:
    
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + \
        (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
        (1 - self.rho) * layer.dbiases**2
        layer.weights += -self.current_learning_rate * \
        layer.dweights / \
        (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
        layer.dbiases / \
        (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
            
class Optimizer_Adam():
    def __init__(self,learning_rate=1.0,decay=0.,beta1=0.,beta2=0.,epsilon=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate 
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iteration = 0
        
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = \
                self.learning_rate*(1./(1.+self.decay*self.iteration))
    def update_parm(self,layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
        layer.weight_momentum = (self.beta1*layer.weight_momentum)+\
            ((1-self.beta1)*layer.dweights)
        layer.bias_momentum = (self.beta1*layer.bias_momentum)+\
            ((1-self.beta1)*layer.dbiases)
        corrected_weight_momentum = layer.weight_momentum/\
            (1-self.beta1**(self.iteration+1))
        corrected_bias_momentum = layer.bias_momentum/\
            (1-self.beta1**(self.iteration+1))
        layer.weight_cache = (self.beta2*layer.weight_cache)+\
            ((1-self.beta2)*layer.dweights**2)
        layer.bias_cache = (self.beta2*layer.bias_cache)+\
            ((1-self.beta2)*layer.dbiases**2)
        corrected_weight_cache = layer.weight_cache/\
            (1-self.beta2**(self.iteration+1))
        corrected_bias_cache = layer.bias_cache/\
            (1-self.beta2**(self.iteration+1))
                       
        layer.weights += -self.current_learning_rate*corrected_weight_momentum/\
            (np.sqrt(corrected_weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate*corrected_bias_momentum/\
            (np.sqrt(corrected_bias_cache)+self.epsilon)
    def post_update(self):
        self.iteration+=1
