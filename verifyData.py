import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

"""
	Train set (0 a 59999):
		0: 0 a 5922
		1: 5923 a 12664
		2: 12665 18622
		3: 18623 24753
		4: 24754 30595
		5: 30596 36016
		6: 36017 41934
		7: 41935 48199
		8: 48200 54050
		9: 54051 59999
	
	Test set (60000 a 69999):
		0: 60000 a 60979
		1: 60980 a 62114
		2: 62115 63146
		3: 63147 64156
		4: 64157 65138
		5: 65139 66030
		6: 66031 66988
		7: 66989 68016
		8: 68017 68990
		9: 68991 69999
"""

# Carregar o Dataset
mnist = loadmat("mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]

# Imprime qual o intervalo de cada número no Dataset
list_train = [ [] for _ in range(10) ]
list_test = [ [] for _ in range(10) ]

for i in range(70000):
	if i < 60000:
		list_train[int(mnist_label[i])].append(i)
	else:
		list_test[int(mnist_label[i])].append(i)

print("Train set (0 a 59999):")
for i in range(10):
	print(f"	{i}: {min(list_train[i])} a {max(list_train[i])}")
	
print("Test set (0 a 59999):")
for i in range(10):
	print(f"	{i}: {min(list_test[i])} a {max(list_test[i])}")

# Imprimir imagem selecionada
select = 0
img = mnist_data[select].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
