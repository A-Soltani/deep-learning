import copy, numpy as np
from abc import ABC, abstractmethod

np.random.seed(0)

class dataset:
    def __init__(self, binary_dim):
        # creating lookup table for converting int to binary
        self.int2binary = {}
        
        self.largest_number = pow(2,binary_dim)
        range_numbers = range(self.largest_number)
        
        # genrating corresponding binary array
        # for example binary[0] = array([0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)
        binary = np.unpackbits(np.array([range_numbers],dtype=np.uint8).T,axis=1)
        
        # adding binary array to int2binary (lookup table)
        for i in range_numbers:
            self.int2binary[i] = binary[i]
    
    # generate a sample addition problem (a + b = c)
    def get_sample_addition_problem(self):
        a_int = np.random.randint(self.largest_number/2) # int version # generate random int between [1,largest_number/2)
        a = self.int2binary[a_int] # binary encoding

        b_int = np.random.randint(self.largest_number/2) # int version
        b = self.int2binary[b_int] # binary encoding

        # true answer => summation
        c_int = a_int + b_int
        c = self.int2binary[c_int]

        return a, b, c, a_int, b_int, c_int

class activation(ABC):
    
    @abstractmethod
    def forward(self, net):
        pass
    
    @abstractmethod
    def backward(self, output):
        pass

class sigmoid_activation(activation):
        
    def forward(self, net):
        return 1/(1 + np.exp(-net))
    
    def backward(self, output):
        return output*(1 - output)

class network_layer(ABC):
    
    def __init__(self, neuron_count):
        self.neuron_count = neuron_count

class input_layer(network_layer): 
    
    def forward(self, X, W_input):
        return np.dot(X,W_input)

class hiddenLayerUnfold:
    
    def __init__(self, neuron_count):
        
        # Save the values obtained at Hidden Layer of current state in a list to keep track
        self.hidden_layer_values  = list()
        
        # Initially, there is no previous hidden state. So append "0" for that
        self.hidden_layer_values.append(np.zeros(neuron_count))
    
    def save_previous_hidden_layer_value(self, previous_hidden_layer_value):
        self.hidden_layer_values.append(copy.deepcopy(previous_hidden_layer_value))

class hidden_layer(network_layer):
    
    def __init__(self, neuron_count):
        super().__init__(neuron_count)
        #self.hiddenLayerUnfold = hiddenLayerUnfold(neuron_count)
        # Save the values obtained at Hidden Layer of current state in a list to keep track
        self.hidden_layer_values  = list()
        
        # Initially, there is no previous hidden state. So append "0" for that
        self.hidden_layer_values.append(np.zeros(neuron_count))
    
    def forward(self, input_layer_output, W_hidden):
        prev_hidden = self.hidden_layer_values[-1]      
        net_hidden = input_layer_output + np.dot(prev_hidden, W_hidden)
        sigmoid = sigmoid_activation()
        return sigmoid.forward(net_hidden)
    
    def save_previous_hidden_layer_value(self, previous_hidden_layer_value):
        #self.hiddenLayerUnfold.save_previous_hidden_layer_value(previous_hidden_layer_value)
        self.hidden_layer_values.append(copy.deepcopy(previous_hidden_layer_value))

class output_layer(network_layer):
    
    def forward(self, hidden_layer_output, W_output):
        net_output = np.dot(hidden_layer_output, W_output)
        sigmoid = sigmoid_activation()
        return sigmoid.forward(net_output)

class weight:
    
    @staticmethod
    def GetWeightMatrix(first_dimension, second_dimension):
        return 2*np.random.random((first_dimension,second_dimension)) - 1
    
class loss_function():
    
    @staticmethod
    def mse(target_value, predicted_value):
        return np.mean((target_value - predicted_value)**2)

class utility:
    
    @staticmethod
    def print_result(overallError, a_int, b_int, c, d):    
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")

class simple_binary_addition_rnn:
    
    def __init__(self, binary_dim, hidden_dimension, learning_rate):
        
        self.binary_dim = binary_dim
        input_dimension = 2
        output_dimension = 1    
        
        # layers
        self.input_layer = input_layer(input_dimension)
        self.hidden_layer = hidden_layer(hidden_dimension)
        self.output_layer = output_layer(output_dimension)
        
        # initialize weights
        self.W_input = weight.GetWeightMatrix(input_dimension, hidden_dimension)
        self.W_hidden = weight.GetWeightMatrix(hidden_dimension, hidden_dimension)
        self.W_output = weight.GetWeightMatrix(hidden_dimension, output_dimension)
        
        self.learning_rate = learning_rate
        self.overallError = 0
        
        # update values for weights
        self.output_layer_deltas = list()
        
    def feed_forward(self, a, b, c):
        
         # Array to save predicted outputs (binary encoded)
        d = np.zeros_like(c)
    
        # position: location of the bit amongst binary_dim-1 bits; for example, starting point "0"; "0 - 7"
        for position in range(self.binary_dim):

            location = self.binary_dim - position - 1
            X = np.array([[a[location], b[location]]])

            # Actual value for (a+b) = c, c is an array of 8 bits, so take transpose to compare bit by bit with X value.        
            target = np.array([[c[location]]]).T            
            
            # ----------- forward ---------------
            # input_layer forward
            input_layer_output = self.input_layer.forward(X, self.W_input)
            
            
            # hidden_layer forward
            hidden_layer_output = self.hidden_layer.forward(input_layer_output, self.W_hidden)
            
            # self.output_layer.forward
            # predicated_value is a "guess" for each input matrix. 
            # We can now compare how well it did by subtracting the true answer (y) from the guess (predicated_value). 
            predicated_value = self.output_layer.forward(hidden_layer_output, self.W_output)            
        
            # Save the hidden layer to be used later            
            self.hidden_layer.save_previous_hidden_layer_value(hidden_layer_output)

            # Round off the values to nearest "0" or "1" and save it to a list
            d[location] = np.round(predicated_value[0][0])   
            
        return d
    
    
    def train(self, epochs_count):
        
        data = dataset(self.binary_dim)        
        
        
        # This for loop "iterates" multiple times over the training code to optimize our network to the dataset.
        for epoch in range(epochs_count):
            
            overallError = 0
            
            # sample a + b = c
            # for example: 2 + 3 = 5 => (a) 00000010 + (b) 00000011 = (c) 00000101
            a, b, c, a_int, b_int, c_int = data.get_sample_addition_problem()
            
            # where we'll store our best guess (binary encoded)
            # desired predictions => d
            d = np.zeros_like(c)  
            
            d = self.feed_forward(a, b ,c)
            
            #back_propagating(a, b)
    
            # Print out the Progress of the RNN
            if (epoch % 10 == 0):
                utility.print_result(overallError, a_int, b_int, c, d)
                
    
def main():
    binary_dim = 8
    hidden_dimension = 16
    learning_rate = 0.1
    rnn = simple_binary_addition_rnn(binary_dim, hidden_dimension, learning_rate)
    rnn.train(1000)
    
if __name__ == "__main__":
    main()
    