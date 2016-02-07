from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Neuralnet:
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def __init__(self, neurons):
		self.weights = []
		self.outputs = []
		self.inputs = []
		self.errors = []
		self.offsets = []
		self.rate = .1
		for layer in range(len(neurons)-1):
			self.weights.append(
				np.random.normal(
					scale=1/np.sqrt(neurons[layer]), 
					size=[neurons[layer], neurons[layer + 1]]
					)
				)
			self.outputs.append(np.empty(neurons[layer]))
			self.inputs.append(np.empty(neurons[layer]))
			self.errors.append(np.empty(neurons[layer]))
			self.offsets.append(np.random.normal(scale=1/np.sqrt(neurons[layer]), size=neurons[layer + 1]))
		self.inputs.append(np.empty(neurons[-1]))
		self.errors.append(np.empty(neurons[-1]))

	def feedforward(self, inputs):
		self.inputs[0] = inputs
		for layer in range(len(self.weights)):
			# self.outputs[layer] = np.tanh(self.inputs[layer])
			self.outputs[layer] = np.vectorize(self.sigmoid)(self.inputs[layer])
			self.inputs[layer + 1] = self.offsets[layer] + np.dot(self.weights[layer].T, self.outputs[layer])

	def backpropagate(self, targets):
		self.errors[-1] = self.inputs[-1] - targets
		for layer in reversed(range(len(self.errors) - 1)):
			# gradient = 1 - self.outputs[layer] * self.outputs[layer]
			gradient = self.outputs[layer] * (1 - self.outputs[layer])
			self.errors[layer] = gradient * np.dot(self.weights[layer], self.errors[layer + 1])
		for layer in range(len(self.weights)):
			self.weights[layer] -= self.rate * np.outer(self.outputs[layer], self.errors[layer + 1])
			self.offsets[layer] -= self.rate * self.errors[layer + 1]

def xor_example():
	net = Neuralnet([2, 2, 1])
	for step in range(10000):
		net.feedforward([0, 0])
		net.backpropagate([0])
		net.feedforward([0, 1])
		net.backpropagate([1])
		net.feedforward([1, 0])
		net.backpropagate([1])
		net.feedforward([1, 1])
		net.backpropagate([0])
	net.feedforward([0, 0])
	print net.inputs[-1]
	net.feedforward([0, 1])
	print net.inputs[-1]
	net.feedforward([1, 0])
	print net.inputs[-1]
	net.feedforward([1, 1])
	print net.inputs[-1]

def identity_example():
	net = Neuralnet([1, 1, 1])
	for step in range(10000):
		x = np.random.uniform(0, 1)
		net.feedforward([x])
		net.backpropagate([x])
	net.feedforward([.25])
	print net.inputs[-1]



# Create neural network that accepts a 28 by 28 pixel array
net = Neuralnet([28 * 28, 28, 10])

# Extract training data from files
data = []
for i in range(10):
	with open('digits/digits' + str(i), 'r') as f:
		data.append(np.fromfile(f, dtype=np.uint8).reshape(1000, 28, 28))

# Train neural network using training data
for epoch in range(10):
	for sample in np.random.permutation(1000):
		for digit in np.random.permutation(10):
			inputs = data[digit][sample].flatten()
			net.feedforward(inputs)
			targets = np.zeros(10)
			targets[digit] = 1
			net.backpropagate(targets)
	print 'Epoch ' + str(epoch) + ' complete'

if len(sys.argv) > 1:
	# Use image specified by user
	image = Image.open(sys.argv[1])
	image = image.resize((28, 28))
	image = np.asarray(image)
	image = image[:, :, 0]
else:
	# Use random image from training data
	digit = np.random.randint(10)
	sample = np.random.randint(1000)
	image = data[digit][sample]

# Show image being fed into neural network
plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
plt.show()

# Feed image into neural network
net.feedforward(image.flatten())

# Print neural network outputs and classification
print net.inputs[-1]
print np.argmax(net.inputs[-1])