import numpy as np
import matplotlib.pyplot as plt

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
	print(net.inputs[-1])
	net.feedforward([0, 1])
	print(net.inputs[-1])
	net.feedforward([1, 0])
	print(net.inputs[-1])
	net.feedforward([1, 1])
	print(net.inputs[-1])

def identity_example():
	net = Neuralnet([1, 1, 1])
	for step in range(10000):
		x = np.random.uniform(0, 1)
		net.feedforward([x])
		net.backpropagate([x])
	net.feedforward([.25])
	print(net.inputs[-1])



data = []
for i in range(10):
	with open('data/data' + str(i), 'r') as f:
		data.append(np.fromfile(f, dtype=np.uint8).reshape(1000, 28, 28))

plt.imshow(data[8][99], cmap='gray', vmin=0, vmax=255, interpolation='none')
plt.show()




