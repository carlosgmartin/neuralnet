import numpy as np
import matplotlib.pyplot as plt

class Neuralnet:
	def __init__(self, neurons):
		self.layers = len(neurons)

		# Learning rate
		self.rate = .01

		# Input vectors
		self.inputs = []
		# Output vectors
		self.outputs = []
		# Error vectors
		self.errors = []
		# Weight matrices
		self.weights = []
		# Bias vectors
		self.biases = []

		for layer in range(self.layers):
			# Create the input, output, and error vector
			self.inputs.append(np.empty(neurons[layer]))
			self.outputs.append(np.empty(neurons[layer]))
			self.errors.append(np.empty(neurons[layer]))

		for layer in range(self.layers - 1):
			# Create the weight matrix
			self.weights.append(np.random.normal(
				scale=1.0/np.sqrt(neurons[layer]),
				size=[neurons[layer], neurons[layer + 1]]
			))
			# Create the bias vector
			self.biases.append(np.random.normal(
				scale=1.0/np.sqrt(neurons[layer]),
				size=neurons[layer + 1]
			))

	def feedforward(self, inputs):
		# Set input neuron inputs
		self.inputs[0] = inputs
		for layer in range(self.layers - 1):
			# Find output of this layer from its input
			self.outputs[layer] = np.tanh(self.inputs[layer])
			# Find input of next layer from output of this layer and weight matrix (plus bias)
			self.inputs[layer + 1] = np.dot(self.weights[layer].T, self.outputs[layer]) + self.biases[layer]
		self.outputs[-1] = np.tanh(self.inputs[-1])

	def backpropagate(self, targets):
		# Calculate error at output layer
		self.errors[-1] = self.outputs[-1] - targets
		# Calculate error vector for each layer
		for layer in reversed(range(self.layers - 1)):
			gradient = 1 - self.outputs[layer] * self.outputs[layer]
			self.errors[layer] = gradient * np.dot(self.weights[layer], self.errors[layer + 1])
		# Adjust weight matrices and bias vectors
		for layer in range(self.layers - 1):
			self.weights[layer] -= self.rate * np.outer(self.outputs[layer], self.errors[layer + 1])
			self.biases[layer] -= self.rate * self.errors[layer + 1]

# Create a neural network that accepts a 28 by 28 array as input and has 10 output neurons
net = Neuralnet([28 * 28, 200, 10])

# Extract handwritten digits from files
digits = []
for digit in range(10):
	with open('digits/' + str(digit), 'r') as digitfile:
		digits.append(np.fromfile(digitfile, dtype=np.uint8).reshape(1000, 28, 28))

# Train neural network on entire data set multiple times
for epoch in range(10):
	# Total error for this epoch
	error = 0
	# Choose a sample index
	for sample in np.random.permutation(1000):
		# Choose a digit
		for digit in np.random.permutation(10):
			# Extract input data
			inputs = digits[digit][sample].flatten()
			# Feed input data to neural network
			net.feedforward(inputs)
			# Target output consists of -1s except for matching digit
			targets = np.full(10, -1, dtype=np.float32)
			targets[digit] = 1
			# Train neural network based on target output
			net.backpropagate(targets)
			error += np.sum(net.errors[-1] * net.errors[-1])
	print 'Epoch ' + str(epoch) + ' error: ' + str(error)

while True:
	inputstring = raw_input('Please input a digit: ')
	if inputstring.isdigit():
		digit = int(inputstring)
		if digit in range(10):
			# Choose a random sample
			sample = np.random.randint(1000)
			image = digits[digit][sample]

			# Show image being fed into neural network
			plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
			plt.show()

			# Feed image into neural network
			net.feedforward(image.flatten())

			# Print neural network outputs and classification
			print 'Classification: ' + str(np.argmax(net.outputs[-1]))






