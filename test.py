import numpy as np
from data import training_set, test_set

# nrOfColumns = 3
# nrOfRows = 3
# nrOfSymbols = 2

class Node:
    def __init__(self):
        # self.input_sum = 0.0
        # self.output_value = 0.0
        self.value = 0
        self.links = []

    def getValue(self):
        self.value = sum(link.getValue() for link in self.links)
        return self.value

class Link:
    def __init__(self, weight, inNode, outNode):
        self.weight = weight
        self.inNode = inNode
        # self.outNode = outNode
        outNode.links.append(self)

    def getValue(self):
        value = self.weight * self.inNode.getValue()
        return value

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        # Maak een lijst met input nodes
        self.inputNodes = [Node() for i in range(input_size)]
        # Maak een lijst met output nodes
        self.outputNodes = [Node() for i in range(output_size)]

        links = []

        for inputNode in self.inputNodes:
            for outputNode in self.outputNodes:
                links.append(Link(inputNode, outputNode))

        # self.weights = []
        # for _ in range(output_size):
        #     output_weights = []
        #     for _ in range(input_size):
        #         # weight = np.random.normal()
        #         weight = Link(np.random.normal())
        #         output_weights.append(weight)
        #     self.weights.append(output_weights)
        print("======")

        print(len(self.weights))
        print(len(self.weights[0]))
        
        print("======")
    
    def forward_propagation(self, inputs):
        for i in range(len(self.inputNodes)):
            self.inputNodes[i].output_value = inputs[i]
        
        for j in range(len(self.outputNodes)):
            node = self.outputNodes[j]
            node.input_sum = 0.0
            for i in range(len(self.inputNodes)):
                node.input_sum += self.inputNodes[i].output_value * self.weights[j][i].weight
            node.output_value = node.input_sum
        
        total_output = sum(np.exp(node.output_value) for node in self.outputNodes)
        for node in self.outputNodes:
            node.output_value = np.exp(node.output_value) / total_output
    
    def backward_propagation(self, targets, learning_rate):
        for j in range(len(self.outputNodes)):
            output_node = self.outputNodes[j]
            output_error = targets[j] - output_node.output_value
            for i in range(len(self.inputNodes)):
                delta_weight = learning_rate * output_error * self.inputNodes[i].output_value
                self.weights[j][i].weight += delta_weight
    
    def train(self, training_set, learning_rate, epochs):
        for _ in range(epochs):
            for example, target in training_set:
                inputs = np.array(example).flatten()
                self.forward_propagation(inputs)


                targets = [0.0, 0.0]
                if target == 'O':
                    targets[0] = 1.0
                else:
                    targets[1] = 1.0
                self.backward_propagation(targets, learning_rate)
    
    def predict(self, inputs):
        self.forward_propagation(inputs)
        return [node.output_value for node in self.outputNodes]

# Training data
# training_set = [
#     ([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'O'),
#     ([[1, 1, 0], [1, 0, 1], [0, 1, 1]], 'X')
# ]

# Test data
# test_set = [
#     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#     [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
# ]

# Create and train the neural network
input_size = 9
output_size = 2
learning_rate = 0.01
epochs = 1000

nn = NeuralNetwork(input_size, output_size)
nn.train(training_set, learning_rate, epochs)

# Test the neural network
for example in test_set:
    inputs = example[0][0]+ example[0][1] +example[0][2]
    # inputs = np.array(example).flatten()
    prediction = nn.predict(inputs)
    print("Prediction:", prediction)
    if prediction[0] > prediction[1]:
        print("It's a circle (O)")
    else:
        print("It's a cross (X)")
    print()
