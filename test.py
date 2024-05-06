import numpy as np
import random
from data import training_set, test_set

once = 0

# nrOfColumns = 3
# nrOfRows = 3
# nrOfSymbols = 2

# global once
# if once == 0:
#     # for link in links:
#     #     print(link.inNode.value)
#     for node in inputNodes:
#         print(node.value)
#     once = 1
#     print("===========")

class Node:
    def __init__(self):
        # self.input_sum = 0.0
        # self.output_value = 0.0
        self.value = 0
        self.links = []

    def getValue(self):
        if(len(self.links)==0):
            return self.value
        
        self.value = sum(link.getValue() for link in self.links)
        return self.value

class Link:
    def __init__(self, inNode, outNode):
        # self.weight = random.random()
        self.weight = 1
        self.inNode = inNode
        # self.outNode = outNode
        outNode.links.append(self)

    def getValue(self):
        value = self.weight * self.inNode.getValue()
        return value

def train(learning_rate, epochs):
    for i in range(epochs):
        for example, target in training_set:
            inputs = np.array(example).flatten()
            forward_propagation(inputs)

            targets = [0.0, 0.0]
            if target == 'O':
                targets[0] = 1.0
            else:
                targets[1] = 1.0
            backward_propagation(targets, learning_rate)

def forward_propagation(inputs):      
    # Assign input values from trainingset to inputnodes
    for i in range(input_size):
        inputNodes[i].value = inputs[i]
    
def backward_propagation(targets, learning_rate):
    # Adjust weights of every link
    for j in range(output_size):
        output_node = outputNodes[j]
        output_error = targets[j] - output_node.getValue()
        for link in output_node.links:
            delta_weight = learning_rate * output_error * link.inNode.getValue()
            link.weight += delta_weight

def predict(inputs):
    forward_propagation(inputs)
    # Apply softmax function
    total_output = sum(np.exp(node.getValue()) for node in outputNodes)
    return [np.exp(node.getValue()) / total_output for node in outputNodes]

# Create and train the neural network
input_size = 9
output_size = 2
learning_rate = 0.01
epochs = 1_000

inputNodes = [Node() for i in range(input_size)]

outputNodes = [Node() for i in range(output_size)]


links = []

for inputNode in inputNodes:
    for outputNode in outputNodes:
        links.append(Link(inputNode, outputNode))

train(learning_rate, epochs)


# Test the neural network
for example in test_set:
    inputs = example[0][0]+ example[0][1] +example[0][2]
    inputs = np.array(example[0]).flatten()
    print("Input:", inputs)
    
    prediction = predict(inputs)
    print("Prediction:", prediction)
    print("Expected:", example[1])

    if prediction[0] > prediction[1]:
        print("It's a circle (O)")
    else:
        print("It's a cross (X)")
    print()
