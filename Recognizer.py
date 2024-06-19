# ========================== Bibliotecas ==============================
from scipy.io import loadmat 
import numpy as np 
from scipy.optimize import minimize 

# ============================= Funções ===============================
def predict(W, data): 
	#	Preve o número a partir do Weight inserido
	m = data.shape[0] 
	ones = np.ones((m, 1))
	
	for i in range(W.shape[0]):
		data = np.append(ones, data, axis=1)
		value = np.dot(data, W[i].transpose())
		data = 1 / (1 + np.exp(-value))
		 
	p = (np.argmax(data, axis=1))
	return p 

# ============================ Modelo ==================================

def neural_network(params, input_size, hidden_size, output_size, data, answer):
	# inicializar
	layer = 3
	W = np.empty(layer, dtype=object) 
	W[0] = np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)) # shape = (38, 785) 
	W[1] = np.reshape(params[hidden_size * (input_size + 1):hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1)], (hidden_size, hidden_size + 1)) # shape = (38, 39) 
	W[2] = np.reshape(params[hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1):], (output_size, hidden_size + 1)) # shape = (10, 39)
	
	# Forward propagation
	x = np.empty(layer+1, dtype=object)
	s = np.empty(layer+1, dtype=object)
	
	m = data.shape[0]
	bias = np.ones((m, 1))
	
	x[0] = data
	for i in range(layer):
		x[i] = np.append(bias, x[i], axis=1)
		s[i+1] = np.dot(x[i], W[i].T)
		x[i+1] = 1 / (1 + np.exp(-s[i+1]))

	# Calcular o custo
	ans_vect = np.zeros((m, 10)) 
	for i in range(m): 
		ans_vect[i, int(answer[i])] = 1
	J = (1 / m) * (1 / 2) * (np.sum(np.sum((x[layer] - ans_vect) * (x[layer] - ans_vect))))

	# Backpropagation
	delta = np.empty(layer+1, dtype=object) 
	delta[layer] = x[layer] - ans_vect
	for i in range(layer-1, 0, -1):
		delta[i] = np.dot(delta[i+1], W[i]) * x[i] * (1 - x[i])
		delta[i] = delta[i][:, 1:]

	# Gradient
	W_grad = np.empty(layer, dtype=object) 
	for i in range(layer):
		W[i][:, 0] = 0
		W_grad[i] = (1 / m) * np.dot(delta[i+1].T, x[i])
	grad = np.concatenate((W_grad[0].flatten(),W_grad[1].flatten(),W_grad[2].flatten())) 

	return J, grad 

# ============================ Main ===============================
# Carregar o Dataset
mnist = loadmat("mnist-original.mat")
mnist_data = mnist["data"].T / 255
mnist_label = mnist["label"][0]

# Dividir os dados em dados de treino e de teste
mnist_data_train = mnist_data[:60000, :]
mnist_data_test = mnist_data[60000:, :]
mnist_label_train = mnist_label[:60000]
mnist_label_test = mnist_label[60000:]

input_size = 784 # Imagem de 28x28 tem 784 pixels
hidden_size = 38 # Tamanho das duas camadas ocultas
output_size = 10 # Números de 0 a 9

# Inicializar os weights iniciais aleatoriamente entre [-epsilon, epsilon)
epsilon = 0.15
Weight1 = np.random.rand(hidden_size, input_size+1) * (2 * epsilon) - epsilon
Weight2 = np.random.rand(hidden_size, hidden_size+1) * (2 * epsilon) - epsilon
Weight3 = np.random.rand(output_size, hidden_size+1) * (2 * epsilon) - epsilon

# Coloca os paramentos em uma só
params = np.concatenate((Weight1.flatten(), Weight2.flatten(), Weight3.flatten()))
myargs = (input_size, hidden_size, output_size, mnist_data_train, mnist_label_train) 

# Minimizar o erro da função "neural_network"
results = minimize(neural_network, x0=params, args=myargs, options={'maxiter': 100}, method="L-BFGS-B", jac=True) 
solution = results["x"]

# Extrair a resposta
W = np.empty(3, dtype=object)
W[0] = np.reshape(solution[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)) # shape = (38, 785) 
W[1] = np.reshape(solution[hidden_size * (input_size + 1):hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1)], (hidden_size, hidden_size + 1)) # shape = (38, 39) 
W[2] = np.reshape(solution[hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1):], (output_size, hidden_size + 1)) # shape = (10, 39) 

# Checking test set accuracy of our model 
pred = predict(W, mnist_data_test) 
print('Test Set Accuracy: {:f}'.format((np.mean(pred == mnist_label_test) * 100))) 

# Checking train set accuracy of our model 
pred = predict(W, mnist_data_train) 
print('Training Set Accuracy: {:f}'.format((np.mean(pred == mnist_label_train) * 100)))
