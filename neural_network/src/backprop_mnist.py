#Simple neural network implementation using Sigmoid as the activation function

from numpy import exp, array, random, dot, rint, power, ones
from numpy import concatenate, load
import mnist_loader
training_data, validation_data, test_data= \
mnist_loader.load_data_wrapper()

#syntax: NeuralNetwork([5,4,4,2])
## 1 input layers with 5 neurons
## 2 hidden layer with 4 neurons each
## 1 output layer with 2 neurons

class NeuralNetwork:
    def __init__(self, sizes):
        #random.seed(1)
	self.num_layers = len(sizes) #includes input+output
	self.sizes = sizes
	self.weights = []
	self.biases = []
	self.activation = [] #to store activation per layer,per epoch, for backprop
	self.weighted_input = [] #to store weighted input per layer, per epoch
	self.layer_error=[]
	for i in xrange(0,self.num_layers-1):
	   self.weights.append((2*random.random((sizes[i+1],sizes[i]))-1).T)
           self.biases.append((2*random.random((1,sizes[i+1]))-1))
	   #self.weights.append(random.randn(sizes[i+1],sizes[i]).T)
	   #self.biases.append(random.randn(1,sizes[i+1]))

    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x) #sigmoid(1-sigmoid)

    def forward_pass(self, inputs):
        self.activation=[] # "a" matrix, empty at beginning of each epoch
	self.weighted_input=[] # "z" matrix, empty at beginning of each epoch
	a=inputs #initial a = input neurons, a gets updated every layer
	for i in xrange(0,self.num_layers-1):
	   z=dot(a, self.weights[i]) + self.biases[i]
	   a=self.sigmoid(z)
           self.activation.append(a) #store activation per layer for backprop 
           self.weighted_input.append(z) #store weighted input per layer for backprop
        return a #return the last activation (last layer activation)

    def train(self, training_set_inputs, training_set_outputs, epochs, learning_rate):
        n=len(training_set_inputs)
	for iteration in xrange(0,epochs):
            output = self.forward_pass(training_set_inputs) #forward pass for each epoch
	    error = output - training_set_outputs #error in last layer (aL - y)
	    #cost = 0.5 * power(error,2) #cost function SSD
	    zL=self.weighted_input[self.num_layers-2]
	    sig=self.sigmoid(zL)
	    d_sig=self.sigmoid_derivative(sig)
	    output_error=error*d_sig # "dL" matrix

	    #BACK PROPAGATION
            self.layer_error=[] #make sure it's empty before appending
	    for i in xrange(0,self.num_layers-1):
                if i==0: #for last layer (l=L)
		   self.layer_error.append(output_error)
		else: #for l=L-1,L-2,....until l=2
		   zl=self.weighted_input[self.num_layers-2-i]
		   sig=self.sigmoid(zl)
		   d_sig=self.sigmoid_derivative(sig)
		   d=dot(self.layer_error[i-1],self.weights[self.num_layers-1-i].T)*d_sig
		   self.layer_error.append(d)

	    #reverse order of self.layer_error (from l=2 to l=L)
	    self.layer_error=list(reversed(self.layer_error))

	    #GRADIENT DESCENT
	    for i in xrange(self.num_layers,1,-1):
		if i==2:
		   weight_error_term=dot(training_set_inputs.T,self.layer_error[i-2])
		   bias_error_term=dot(ones(n),self.layer_error[i-2])
		else:
		   weight_error_term=dot(self.activation[i-3].T,self.layer_error[i-2])
		   bias_error_term=dot(ones(n),self.layer_error[i-2])
		self.weights[i-2] = self.weights[i-2] - ((learning_rate*weight_error_term)/n)
		self.biases[i-2] = self.biases[i-2] - ((learning_rate*bias_error_term)/n)

    def predict(self, inputs):
        a=inputs
	for i in xrange (0,self.num_layers-1):
           z=dot(a, self.weights[i]) + self.biases[i]
           a=self.sigmoid(z)
        return a

if __name__ == "__main__":
    neural_network = NeuralNetwork([784,100,10])
    learning_rate=3.14
    epochs=10

    training_set_inputs = load("train_input.npy")
    training_set_outputs = load("train_output.npy")
    test_set_inputs = load("test_input.npy")
    test_set_outputs = load("test_output.npy")
    print "Neural network: ", neural_network.sizes
    print "Number of  layers: ",neural_network.num_layers
    print "Learning rate: ", learning_rate
    print "Epochs: ", epochs
    print "Training......."
    neural_network.train(training_set_inputs, training_set_outputs, epochs, learning_rate)
    print "TRAINING COMPLETED"

    #Successful percentage
    n=len(test_set_inputs)
    correct_prediction=0
    for i in xrange(0,n):
	results=[[]]
	results=neural_network.predict(test_set_inputs[i])
	results=results[0]
	m=max(results)
	for j in xrange(0,10):
	   if results[j]==m:
		prediction=j
	actual=test_set_outputs[i]
	if int(prediction)==int(actual):
	   correct_prediction+=1

    print "Accuracy: ", 100.0*correct_prediction/n, "%"
