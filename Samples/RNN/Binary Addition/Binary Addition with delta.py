import numpy as np

# importing "collections" for deque operations 
from collections import deque

from abc import ABC, abstractmethod

np.random.seed(0)

class utility:
    
    @staticmethod
    def print_result(a, b, c, predicated_values, epoch):  
        d = np.zeros_like(c)
        for position in range(len(predicated_values)):             
             d[position] = np.round(predicated_values[position][0][0])
        
        print("epoch:", epoch)
        print("a:   " + str(a))
        print("b:   " + str(b))        
        print("c:   " + str(c))
        print("Pred:" + str(d))
        
        print("----------------------")

class dataset:
    @staticmethod
    def get_data(samples_count, binary_dim):
        
        largest_number = pow(2,binary_dim)
        
        samples = list()
        for i in range(samples_count):
            
            a = np.random.randint(largest_number/2) 

            b = np.random.randint(largest_number/2)

            # true answer => summation
            c = a + b
            
                    
            int_array = np.array([[a], [b], [c]], dtype=np.uint8)
            
            binary_array = np.unpackbits(int_array, axis=1)
            
            samples.append(binary_array)
            
        return samples

class random_generator:
    
    @staticmethod
    def get_random_weight_matrix(input_dimension, output_dimension):
        return 2*np.random.random((input_dimension,output_dimension)) - 1

class multiply_gate:
    
    @staticmethod
    def forward(inputs, weights):
        return np.dot(inputs, weights)
    
    @staticmethod
    def backward(weights):
        return weights.T

class add_gate:
    
    @staticmethod
    def forward(input1, input2):
        return input1 + input2
    
    @staticmethod
    def backward(input1, input2):
        return input1.backward() + input2.backward()


class sigmoid_activation():
        
    @staticmethod
    def forward(net):
        return 1/(1 + np.exp(-net))
    
    @staticmethod
    def backward(output):
        return output*(1 - output)      


class network_layer(ABC):
    
    def __init__(self, input_dimension, output_dimension):        
        self.weights = random_generator.get_random_weight_matrix(input_dimension, output_dimension)
    
    @abstractmethod
    def forward(self, input):
        pass
    
    
class input_layer(network_layer):
    
    def forward(self, X):
        return multiply_gate.forward(X, self.weights)
    
    def backward(self):
        return multiply_gate.backward(self.weights)
        
class hidden_layer(network_layer):
    
    def forward(self, net_input, s_t_prev):        
        net_hidden = add_gate.forward(net_input, multiply_gate.forward(s_t_prev, self.weights))
        return sigmoid_activation.forward(net_hidden)
    
class output_layer(network_layer):
    
    def forward(self, activation_hidden): 
        net_output = multiply_gate.forward(activation_hidden, self.weights)    
        return sigmoid_activation.forward(net_output)

class binary_addition_rnn:
    
    def __init__(self, binary_dim, hidden_dimension, learning_rate):
        
        self.learning_rate = learning_rate
        self.binary_dim = binary_dim
        input_dimension = 2 # two numbers a, b
        self.hidden_dimension = hidden_dimension
        output_dimension = 1 # result of addition, c = a+b
        
        self.input_layer = input_layer(input_dimension, hidden_dimension)
        self.hidden_layer = hidden_layer(hidden_dimension, hidden_dimension)
        self.output_layer = output_layer(hidden_dimension,output_dimension)
        
         # predicated_values array
        self.predicated_values = np.zeros(self.binary_dim)
    
    def feed_forward(self, a, b, c):        
        
        hidden_values = list()
        hidden_values.append(np.zeros((1, self.hidden_dimension)))
        
        prediction_values = deque([])
        
        # Proceed from right-to-left, column-by-column, starting from last digit
        for column in range(self.binary_dim-1, -1, -1):
            
            # It is given two input digits at each time step. 
            X = np.array([[a[column], b[column]]])
            
            # input layer
            net_input_layer = self.input_layer.forward(X) # X*W_in
            
            # hidden layer
            s_t_prev = hidden_values[-1]
            activation_hidden = self.hidden_layer.forward(net_input_layer, s_t_prev)
            
            # save activation_hidden for BPTT
            hidden_values.append(activation_hidden)
            
            # output layer
            prediction_value = self.output_layer.forward(activation_hidden)
            prediction_values.appendleft(prediction_value)
            
        # print(prediction_values)
        return prediction_values, hidden_values
            
    def bptt(self, a, b, c, predicated_values, hidden_values):        
       
        
        future_delta_net_hidden_explicit = np.zeros(self.hidden_dimension)
        
        # Initialize Updated Weights Values
        W_output_update = np.zeros_like(self.output_layer.weights)
        W_hidden_update = np.zeros_like(self.hidden_layer.weights)
        W_input_update = np.zeros_like(self.input_layer.weights)
        
        for time_step in range(self.binary_dim):
            
            # s_t = h(t)
            time_step_hidden_value_index = self.binary_dim - time_step
            s_t = hidden_values[time_step_hidden_value_index]
            s_t_prev = hidden_values[time_step_hidden_value_index -1]
            
            # target value
            y = np.array([[c[time_step]]]).T
            X = np.array([[a[time_step],b[time_step]]])
            
            y_hat = predicated_values[time_step]   
           
           
            # delta_net_output
            dl_d_y_hat = y_hat-y
            dy_hat_d_net_output = y_hat*(1-y_hat)
            delta_net_output = dl_d_y_hat * dy_hat_d_net_output           
            
            # delta_net_hidden(t)           
            delta_net_hidden = (delta_net_output.dot(self.output_layer.weights.T) + future_delta_net_hidden_explicit.dot(self.hidden_layer.weights.T))* sigmoid_activation.backward(s_t)
            
            # save delta_net_hidden_explicit as future_delta_net_hidden_explicit for next backpropagation step
            future_delta_net_hidden_explicit = delta_net_output.dot(self.output_layer.weights.T) * sigmoid_activation.backward(s_t)
                        
            # Updating all W_output, W_hidden and  W_output                          
            W_output_update += np.atleast_2d(s_t).T.dot(delta_net_output)
            W_hidden_update += np.atleast_2d(s_t_prev).T.dot(delta_net_hidden) 
            W_input_update  += X.T.dot(delta_net_hidden)            

            # # error at output layer
            # outputlayer_error = y - y_hat
            # outputlayer_delta = (outputlayer_error)*sigmoid_activation.backward(y_hat)*(-1)
        
            # # error at hidden layer * sigmoid_derivative(future_hidden)
            # hidden_delta = (future_hidden_delta.dot(self.hidden_layer.weights.T) * sigmoid_activation.backward(future_hidden) + outputlayer_delta.dot(self.output_layer.weights.T)) * sigmoid_activation.backward(s_t)

            # # update all weights 
            # W_output_update += np.atleast_2d(s_t).T.dot(outputlayer_delta)
            # W_hidden_update += np.atleast_2d(s_t_prev).T.dot(hidden_delta) 
            # W_input_update  += X.T.dot(hidden_delta) 
            # future_hidden_delta = hidden_delta
            # future_hidden = s_t 
        
        return W_output_update, W_hidden_update, W_input_update
    
    def back_propagate(self, a, b, c, predicated_values, hidden_values):
        
        W_output_update, W_hidden_update, W_input_update = self.bptt(a, b, c, predicated_values, hidden_values)
        
        self.output_layer.weights -= W_output_update * self.learning_rate
        self.hidden_layer.weights -= W_hidden_update * self.learning_rate
        self.input_layer.weights -= W_input_update * self.learning_rate
        
    
    def train(self, epochs_count):
           
        data = dataset.get_data(epochs_count, self.binary_dim)        
        
        # This for loop "iterates" multiple times over the training code to optimize our network to the dataset.
        for epoch in range(epochs_count):
            
            overallError = 0
            
            # sample a + b = c
            # for example: 2 + 3 = 5 => (a) 00000010 + (b) 00000011 = (c) 00000101            
            sample = data[epoch]
            a = sample[0]
            b = sample[1]
            c = sample[2]
            
            predicated_values, hidden_values = self.feed_forward(a, b, c)
            
            self.back_propagate(a, b, c, predicated_values, hidden_values)
            
            # Print out the Progress of the RNN
            if (epoch % 1000 == 0):
                 utility.print_result(a, b, c, predicated_values, epoch)           
            

if __name__ == '__main__':
    data = dataset.get_data(10000, 8)
    
    rnn = binary_addition_rnn(8, 16, 0.1)
    rnn.train(10000)


