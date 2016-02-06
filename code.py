import numpy as np
import matplotlib.pyplot as plt

data = []
for i in range(10):
	with open('data/data' + str(i), 'r') as f:
		data.append(np.fromfile(f, dtype=np.uint8).reshape(1000, 28, 28))

# plt.imshow(data[8][99], cmap='gray', vmin=0, vmax=255, interpolation='none')
# plt.show()









