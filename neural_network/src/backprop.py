#Simple neural network implementation using Sigmoid as the activation function

from numpy import exp, array, random, dot, rint, power, ones

#syntax: NeuralNetwork([5,4,4,2])
## 1 input layers with 5 neurons
## 2 hidden layer with 4 neurons each
## 1 output layer with 2 neurons
class NeuralNetwork:
    def __init__(self, sizes):
        #random.seed(1)
	self.num_layers = len(sizes) #includes input+output
	print "NUM LAYERS: ",self.num_layers
	self.sizes = sizes
	self.weights = []
	self.biases = []
	self.activation = [] #to store activation per layer,per epoch, for backprop
	self.weighted_input = [] #to store weighted input per layer, per epoch
	self.layer_error=[]
	for i in xrange(0,self.num_layers-1):
	   self.weights.append((2*random.random((sizes[i+1],sizes[i]))-1).T)
           #self.biases.append((2*random.random((1,sizes[i+1]))-1))
	   #self.weights.append(random.randn(sizes[i+1],sizes[i]).T)
	   self.biases.append(random.randn(1,sizes[i+1]))
	
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
	print "n= ",n #number of training data
	for iteration in xrange(0,epochs):
            output = self.forward_pass(training_set_inputs) #forward pass for each epoch
	    error = output - training_set_outputs #error in last layer (aL - y)
	    #cost = 0.5 * power(error,2) #cost function SSD
	    zL=self.weighted_input[self.num_layers-2]
	    sig=self.sigmoid(zL)
	    d_sig=self.sigmoid_derivative(sig)
	    output_error=error*d_sig # "dL" matrix

	    #-------------DEBUG--------------
	    #print "delta:"
            #print error
            #print "dL:"
            #print output_error
            #print "zL:"
            #print zL
	    #print "WEIGHTED INPUT"
	    #print self.weighted_input[self.num_layers-2]
	    #print "Weighted Input size:" 
	    #print len(self.weighted_input)
	    #output_error=dot(error,self.sigmoid_derivative(self.weighted_input[self.num_layers-2].T))
            #print "OUTPUT ERROR (dL)"
            #print output_error
	    #-------------END----------------

	    # * = element wise multiplication
	    # dot = matrix multiplication

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

	    #------------------DEBUG--------------------
	    #print "LAYER_ERROR"
	    #print self.layer_error
	    #print "ACTIVATION"
	    #print self.activation
	    #print "WEIGHTS"
	    #print self.weights
	    #-------------------END---------------------

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

	    #---------------DEBUG----------------------
	    #print "Iteration no.",iteration
	    #print "Updated Weights"
	    #print self.weights
	    #print "Error:"
	    #print error
	    #-----------------END-------------------

    def predict(self, inputs):
        for i in xrange (0,self.num_layers-1):
           z=dot(inputs, self.weights[i]) + self.biases[i]
           a=self.sigmoid(z)
	   inputs=a
        return a

if __name__ == "__main__":
    #neural_network = NeuralNetwork([2,1]) #for doing XOR
    neural_network = NeuralNetwork([3,50,50,2])
    learning_rate=0.15
    epochs=20000
    print "Initial weights:"
    print neural_network.weights
    print "Initial biases:"
    print neural_network.biases

    training_set_inputs = array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    #training_set_inputs = array([[0, 0], [0, 1], [1,0], [1, 1]]) #XOR
    print "Input_set:"
    print training_set_inputs
    
    training_set_outputs = array([[1,0],[0,1],[0,0],[1,1]]) 
    #training_set_outputs = array([[0],[1],[1],[0]]) #XOR
    print "Output_set:"
    print  training_set_outputs
    
    neural_network.train(training_set_inputs, training_set_outputs, epochs, learning_rate)

    print "Adjusted weights:"
    print neural_network.weights

    print "Adjusted biases:"
    print neural_network.biases
    
    #Test
    print "If input [1 0 0], output ", rint(neural_network.predict(array([1, 0, 0])))
    print "If input [0 1 0], output ", rint(neural_network.predict(array([0, 1, 0])))
    print "If input [0 0 1], output ", rint(neural_network.predict(array([0, 0, 1])))
    print "If input [0 1 1], output ", rint(neural_network.predict(array([0, 1, 1])))
    print "If input [1 1 1], output ", rint(neural_network.predict(array([1, 1, 1])))

    # for XOR only
    #print "If input [0 0], output ", rint(neural_network.predict(array([0, 0])))
    #print "If input [0 1], output ", rint(neural_network.predict(array([0, 1])))
    #print "If input [1 0], output ", rint(neural_network.predict(array([1, 0])))
    #print "If input [1 1], output ", rint(neural_network.predict(array([1, 1])))
    
