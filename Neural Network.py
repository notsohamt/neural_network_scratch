import numpy as np

# Initialize parameters
def initialize_parameters(layer_sizes):
    np.random.seed(1)
    parameters = {}
    L = len(layer_sizes)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_sizes[l], 1))
        
    return parameters

# Activation functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def sigmoid_derivative(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

def relu_derivative(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# Forward propagation
def forward_propagation(X, parameters):
    cache = {}
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = relu(Z)
        cache['A' + str(l)] = A
        cache['Z' + str(l)] = Z
    
    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = sigmoid(ZL)
    cache['A' + str(L)] = AL
    cache['Z' + str(L)] = ZL
    
    return AL, cache

# Compute cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
    return np.squeeze(cost)

# Backward propagation
def backward_propagation(AL, Y, cache, parameters):
    grads = {}
    L = len(parameters) // 2
    m = AL.shape[1]
    
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = (cache['A' + str(L-1)], cache['Z' + str(L)])
    grads["dW" + str(L)], grads["db" + str(L)], grads["dA" + str(L-1)] = \
        linear_activation_backward(dAL, current_cache, parameters['W' + str(L)], sigmoid_derivative)
    
    for l in reversed(range(1, L)):
        current_cache = (cache['A' + str(l-1)], cache['Z' + str(l)])
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l)], current_cache, parameters['W' + str(l)], relu_derivative)
        grads["dA" + str(l-1)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp
    
    return grads

def linear_activation_backward(dA, cache, W, activation_backward):
    A_prev, Z = cache
    m = A_prev.shape[1]
    
    dZ = activation_backward(dA, Z)
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    
    return parameters

# Model training
def model(X, Y, layer_sizes, learning_rate=0.01, num_iterations=1000):
    parameters = initialize_parameters(layer_sizes)
    
    for i in range(num_iterations):
        AL, cache = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    return parameters

# Usage example
layer_sizes = [3, 5, 1]  # Example: 3 inputs, 5 hidden units, 1 output
X = np.random.randn(3, 5)  # Example input data
Y = np.array([[1, 0, 1, 0, 1]])  # Example output data

parameters = model(X, Y, layer_sizes, learning_rate=0.01, num_iterations=1000)
