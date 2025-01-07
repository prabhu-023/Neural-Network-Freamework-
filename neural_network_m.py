import numpy as np
import nnfs
from nnfs.datasets import spiral_data,sine_data
import optimizers as op
from visualizations import plot_decision_boundary, plot_training_curves

class LayerInput:
    def forward(self,inputs,training):
        self.output = inputs

class LayerDense:
    nnfs.init()
    def __init__(self,n_inputs,n_neurons,weight_initializer=0.,weight_regularisationl1=0.,bias_regularisationl1=0.,weight_regularisationl2=0.,bias_regularisationl2=0.):
       
        self.weights = weight_initializer*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.weight_regularisationl1 = weight_regularisationl1
        self.bias_regularisationl1 = bias_regularisationl1
        self.weight_regularisationl2 = weight_regularisationl2
        self.bias_regularisationl2 = bias_regularisationl2
          
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = np.dot(self.inputs , self.weights) + self.biases
    def backward(self,dvalues):
        self.dinputs = np.dot(dvalues,self.weights.T)
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues ,keepdims=True,axis=0)
        
        if self.weight_regularisationl1 >0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights <0] = -1
            self.dweights += self.weight_regularisationl1*dl1
        if self.bias_regularisationl1 >0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases <0] = -1
            self.dbiases += self.bias_regularisationl1*dl1
        if self.weight_regularisationl2 >0:
            self.dweights += 2*self.weight_regularisationl2*self.weights
        if self.bias_regularisationl2 >0:
            self.dbiases += 2*self.bias_regularisationl2*self.biases

#Activation ReLU
class Activation_ReLU:
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = np.maximum(0,self.inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0]=0

    def prediction(self,output):
        return output

#Activation Linear
class Activation_liner:

    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = inputs

    def backward(self,dvalues):
        self.dinputs = dvalues

    def prediction(self,output):
        return output

#Softmax Activation function
class Soft_max:
    def forward(self,inpets,training):
        self.inputs = inpets
        exp_values =  np.exp(self.inputs - np.max(self.inputs , axis=1,keepdims=True))
        probabilities = exp_values/np.sum(exp_values , axis=1,keepdims=True)
        self.output = probabilities
        #print(self.outpets[0])
        
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index , (single_output,single_dvalues) in \
            enumerate(zip(self.output,dvalues)):
                single_output = single_output.reshape(-1,1)
                jacobian_matrix = np.diagflat(single_output) - \
                    np.dot(single_output,single_output.T)
                self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

    def prediction(self,output):
        return np.argmax(output,axis=1)



#Dropot Layer
class Dropout:
    """
    Dropout Layer: Randomly sets a fraction of inputs to 0 during training to prevent overfitting.

    Parameters:
        - rate (float): Fraction of inputs to drop (between 0 and 1).
    """
    def __init__(self,rate):
        self.rate = 1-rate
    #forward function
    def forward(self,layer,training):
        self.layer = layer
        if not training:
            self.output = layer.copy()

        else:
            #Create a binary mask of the required rate
            self.binary_mask = np.random.binomial(1,self.rate,size=layer.shape)/\
                self.rate
            self.output = layer*self.binary_mask
    #backward pass
    def backward(self,dvalues):
        self.dinputs = dvalues*self.binary_mask


#Activation Sigmoid
class sigmoid:

    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = 1/(1+np.exp(-self.inputs))

    def backward(self,dvalues):
        self.dinputs = dvalues*self.output*(1-self.output)

    def prediction(self,output):
        return (output>0.5)*1

#Loss Function
class Loss:

   # Set/remember trainable layers
   def remember_trainable_layer(self,trainable_layers):
       self.trainable_layers = trainable_layers


   def loss_regularisation(self):
        regularisation_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularisationl1 > 0:
                regularisation_loss += layer.weight_regularisationl1*np.sum(np.abs(layer.weights))
            if layer.weight_regularisationl2 > 0:
                regularisation_loss += layer.weight_regularisationl2*np.sum(layer.weights**2)
            if layer.bias_regularisationl1 > 0:
                regularisation_loss += layer.bias_regularisationl1*np.sum(np.abs(layer.biases))
            if layer.bias_regularisationl2 > 0:
                regularisation_loss += layer.bias_regularisationl2*np.sum(layer.biases**2)
        return regularisation_loss
    
   def calculate(self,output,y,*,regularization_loss = False):   
        sample_loss = self.forward(output , y)# type: ignore       
        data_loss = np.mean(sample_loss)
        self.accumulated_loss += np.sum(sample_loss)
        self.accumulated_count += len(sample_loss)
        if  not regularization_loss:
            return data_loss,0
        return data_loss , self.loss_regularisation()
   def calculate_accumulated(self,*,regularization_loss = False):

       data_loss = self.accumulated_loss/self.accumulated_count
       if not regularization_loss:
           return data_loss,0
       return data_loss,self.loss_regularisation()
   def new_pass(self):
       self.accumulated_loss=0
       self.accumulated_count=0

#Crossentropy Loss
class CrossEntropy(Loss):
    def forward(self,ypred,ytrue):
        samples = len(ypred)
        ypred_cliped = np.clip(ypred , 1e-7, 1- 1e-7)
        if len(ytrue.shape)==1 :
            correct_confidence=ypred_cliped[range(samples),ytrue]
            #print(correct_confidence[:5])    
            
        elif len(ytrue.shape)==2 :
            correct_confidence = np.sum(ypred_cliped*ytrue ,axis=1)
            
         
        neg_log_form = -np.log(correct_confidence)
        return neg_log_form
       
    def backward(self,dvalues,ytrue):
        samples = len(dvalues)
        level = len(dvalues[0])
        if len(ytrue.shape) == 1:
            ytrue = np.eye(level)[ytrue]
        #clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -ytrue / dvalues
        #self.dinputs = (-ytrue/dvalues)
        self.dinputs = self.dinputs/samples

#Classification Model
class Activation_softmax_Loss_CategoricalCrossentropy():
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs/samples

#Binary LogisticRegression
class Loss_BinaryCrossEntropy(Loss):

    def forward(self,y_pred,y_true):
        y_pred_cliped = np.clip(y_pred,1e-7,1- 1e-7)
        sample_loss = -((y_true*np.log(y_pred_cliped))+\
            ((1-y_pred_cliped)*np.log(1-y_pred_cliped)))
        sample_loss = np.mean(sample_loss,axis=-1)
        return sample_loss

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7) 
        self.dinputs = -(y_true/clipped_dvalues-
                         (1-y_true)/(1-clipped_dvalues))/outputs
        #Normalizing the output
        self.dinputs = self.dinputs/samples



#Loss function
class MeanSquare_error(Loss):

    def forward(self,y_pred,y_true):
        sample_loss = np.mean(np.square(y_true-y_pred),axis=-1)
        return sample_loss

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        output = len(dvalues[0])
        self.dinputs = -2*(y_true-dvalues)/output
        #Normalizing the error
        self.dinputs = self.dinputs/samples

class MeanAbsolute_error(Loss):

    def forward(self,y_pred,y_true):
        sample_loss = np.mean(np.abs(y_true-y_pred),axis=-1)
        return sample_loss
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        output = len(dvalues[0])
        self.dinputs = np.sign(y_true-dvalues)/output
        #Normalizing the error
        self.dinputs = self.dinputs/samples

#Accuracy
class Accuracy:

    def calculate(self,prediction,y_true):
        compairision = self.compare(prediction,y_true)
        accuracy = np.mean(compairision)
        self.accumulated_sum += np.sum(compairision)
        self.accumulated_count += len(compairision)
        return accuracy
    
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum/self.accumulated_count
        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

#Accuracy for classification
class Accuracy_clasification(Accuracy):
    
    def init(self,y_true):
        pass

    def compare(self,y_pred,y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
        return (y_pred==y_true)

#Accuracy for binary-classification
class Accuracy_binary_clasification(Accuracy):
    def init(self,y_true):
        pass

    def compare(self,y_pred,y_true):
        return (y_pred>0.5)*1

#Accuracy for Regression
class Accuracy_regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self,y,reinit=False):
        if (self.precision == None) or reinit:
            self.precision = np.std(y)/250
    def compare(self,y_true,predictions):
        return np.absolute(predictions-y_true)< self.precision

#Model
class Model:
    def __init__(self):
        self.layers = []
        self.softmax_clasifier_output = None

    def add(self,layers):
        self.layers.append(layers)

    def set(self,*,loss,optimizer,accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = LayerInput()
        self.trainable_layers = []
        layers_count = len(self.layers)
        for i in range(layers_count):
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif (0<i<layers_count-1):
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else: 
                self.layers[i].prev = self.layers[i-1] 
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i],'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layer(self.trainable_layers)
        #Setting for Softmax and CrossEntropy
        if isinstance(self.layers[-1],Soft_max) and \
            isinstance(self.loss,CrossEntropy):
            self.softmax_clasifier_output = Activation_softmax_Loss_CategoricalCrossentropy()

    def forward(self,X,training):
        self.input_layer.forward(X,training)

        for layer in self.layers:
            layer.forward(layer.prev.output,training)

        return layer.output

    def backward(self,y_pred,y_true):

        #Adding code for softmax
        if self.softmax_clasifier_output is not None:
            #print("Using combined softmax + cross-entropy backward pass")
            self.softmax_clasifier_output.backward(y_pred,y_true)
            self.layers[-1].dinputs = self.softmax_clasifier_output.dinputs
            self.layers[-1].next = self.softmax_clasifier_output
            #print(self.layers[-1],"\n",self.layers[-1].next)

        else:
            #print("Using standalone loss backward pass")
            self.loss.backward(y_pred, y_true)
            self.layers[-1].dinputs = self.loss.dinputs

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


    def train(self,X,y,*,epochs=1,printevery=1,batch_size=None,validation_data=None):

        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []

        self.accuracy.init(y)
        train_steps = 1
        if validation_data is not None:
            validation_step = 1
            X_val,y_val = validation_data

        if batch_size is not None:
            train_steps = len(X)//batch_size
            if (train_steps*batch_size) < len(X):
                train_steps+=1

            if validation_data is not None:
                validation_step = len(X_val)//batch_size
                if (validation_step*batch_size) < len(X_val):
                    validation_step +=1

        for epoch in range(1,epochs+1):
            
            #Initializing Parameter
            print(f'epoch : {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            #Forward passing 
            for step in range (train_steps):

                if batch_size is None:
                    batch_X = X
                    batch_y = y

                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X,training=True)
                data_loss,regularisation_loss= self.loss.calculate(output,batch_y,regularization_loss=True)
                loss = data_loss+regularisation_loss
                prediction = self.output_layer_activation.prediction(output)
                accuracy = self.accuracy.calculate(prediction,batch_y)
           
                #Backpropagating
                self.backward(output,batch_y)
                #Optimizing the weights
                self.optimizer.pre_update()
                for layer in self.trainable_layers:
                    self.optimizer.update_parm(layer)

                self.optimizer.post_update()

                if (step%printevery==0) or (step == train_steps-1):
                    print(f'step: {step} ',
                            f'accuracy: {accuracy:.3f} ',
                            f'loss: {loss:.3f} ',
                            f'data_loss: {data_loss:.3f} ',
                            f'regularization_loss: {regularisation_loss:.3f}')

            epoch_dloss,epoch_rloss = self.loss.calculate_accumulated(regularization_loss=True)
            epochloss = epoch_dloss+epoch_rloss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(  f'net_accuracy: {epoch_accuracy:.3f} '
                    f'loss: {epochloss:.3f} ',
                    f'data_loss: {epoch_dloss:.3f} ',
                    f'regularization_loss: {epoch_rloss:.3f}')

            train_accuracies.append(epoch_accuracy)
            train_losses.append(epochloss)

        #Validating Data
        if validation_data is not None:

            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(validation_step):
                if batch_size is None:
                    batch_Xv = X_val
                    batch_yv = y_val

                else:
                    batch_Xv = X_val[step*batch_size:(step+1)*batch_size]
                    batch_yv = y_val[step*batch_size:(step+1)*batch_size]


                output_v = self.forward(batch_Xv,training =False)
                loss_v = self.loss.calculate(output_v,batch_yv)
                prediction_v = self.output_layer_activation.prediction(output_v)
                accuracy_v = self.accuracy.calculate(prediction_v,batch_yv)

                

            validation_loss = self.loss.calculate_accumulated() 
            validation_accuracy = self.accuracy.calculate_accumulated()
    

            print(f"validation_step: {step} ",
                    f"accuracy_v: {validation_accuracy:.3f} ",
                    f"loss: {validation_loss[0]:.3f}")

        return train_accuracies, train_losses, val_accuracies, val_losses


'''
#Creating data
X,y = spiral_data(samples=1000,classes=2)
X_test,y_test = spiral_data(samples=100,classes=2)
#ONLY FOR Binary Data
#y =y.reshape(-1,1)
#y_test=y_test.reshape(-1,1)
#############################
model = Model()
#Addinf Layers
model.add(LayerDense(n_inputs=2,n_neurons=128,weight_initializer=0.5,weight_regularisationl2=5e-4,\
    bias_regularisationl2=5e-4))
model.add(Activation_ReLU())
model.add(Dropout(0.1))
model.add(LayerDense(128,3,0.5))
model.add(Soft_max())
#Setting Loss,Accuracy and, Optimiser
model.set(loss=CrossEntropy(),
          optimizer=op.Optimizer_Adam(learning_rate=0.05,decay=5e-5,
                              beta1=0.9,beta2=0.999,epsilon=1e-7),
          accuracy=Accuracy_clasification())
model.finalize()
print(model.layers)
print("Starting training...")
metrics =model.train(X,y,epochs=101,printevery=100,validation_data=(X_test,y_test))
print("Training completed.")

# Extract metrics for plotting
train_accuracies, train_losses, val_accuracies, val_losses = metrics

# Plot training and validation curves
print("\n--- Plotting Training Curves ---")
plot_training_curves(
    epochs=101,
    train_accuracies=train_accuracies,
    train_losses=train_losses,
    val_accuracies=val_accuracies,
    val_losses=val_losses
)

# Visualize decision boundary (for spiral dataset)

plot_decision_boundary(model, X_test, y_test)
'''
