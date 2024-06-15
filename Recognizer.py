# ========================== Bibliotecas ==============================
from scipy.io import loadmat 
import numpy as np 
from scipy.optimize import minimize 

# ============================= Funções ===============================

def initialise(a, b): 
	epsilon = 0.15
	c = np.random.rand(a, b+1) # Inicializa o c como um matrix de tamanho axb+1 de números [0,1)
	c = c * (2 * epsilon) - epsilon # Os valores do c são agora entre [-epsilon, epsilon)
	return c 

def predict(Theta1, Theta2, Theta3, X): 
	m = X.shape[0] 
	one_matrix = np.ones((m, 1)) 
	X = np.append(one_matrix, X, axis=1) # Adding bias unit to first layer 
	z2 = np.dot(X, Theta1.transpose()) 
	a2 = 1 / (1 + np.exp(-z2)) # Activation for second layer 
	one_matrix = np.ones((m, 1)) 
	a2 = np.append(one_matrix, a2, axis=1) # Adding bias unit to hidden layer 
	z3 = np.dot(a2, Theta2.transpose()) 
	a3 = 1 / (1 + np.exp(-z3)) # Activation for third layer 
	one_matrix = np.ones((m, 1)) 
	a3 = np.append(one_matrix, a3, axis=1) # Adding bias unit to hidden layer 
	z4 = np.dot(a3, Theta3.transpose()) 
	a4 = 1 / (1 + np.exp(-z4)) # Activation for third layer 
	p = (np.argmax(a4, axis=1)) # Predicting the class on the basis of max value of hypothesis 
	return p 

# ============================ Modelo ==================================

def neural_network(params, input_size, hidden_size, output_size, data, answer): 
	# Weights are split back to Theta1, Theta2 
	Weight1 = np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)) # shape = (38, 785) 
	Weight2 = np.reshape(params[hidden_size * (input_size + 1):hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1)], (hidden_size, hidden_size + 1)) # shape = (38, 39) 
	Weight3 = np.reshape(params[hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1):], (output_size, hidden_size + 1)) # shape = (10, 39) 

	# Forward propagation 
	m = data.shape[0] 
	one_matrix = np.ones((m, 1)) 
	data = np.append(one_matrix, data, axis=1) # Adding bias unit to first layer 
	a1 = data
	z2 = np.dot(data, Weight1.transpose()) 
	a2 = 1 / (1 + np.exp(-z2)) # Activation for second layer 
	one_matrix = np.ones((m, 1)) 
	a2 = np.append(one_matrix, a2, axis=1) # Adding bias unit to hidden layer 
	z3 = np.dot(a2, Weight2.transpose()) 
	a3 = 1 / (1 + np.exp(-z3)) # Activation for third layer 
	one_matrix = np.ones((m, 1)) 
	a3 = np.append(one_matrix, a3, axis=1) # Adding bias unit to hidden layer 
	z4 = np.dot(a3, Weight3.transpose()) 
	a4 = 1 / (1 + np.exp(-z4)) # Activation for third layer 

	# Changing the y labels into vectors of boolean values. 
	# For each label between 0 and 9, there will be a vector of length 10 
	# where the ith element will be 1 if the label equals i 
	ans_vect = np.zeros((m, 10)) 
	for i in range(m): 
		ans_vect[i, int(answer[i])] = 1

	# Calculating cost function 
	J = (1 / m) * (1 / 2) * (np.sum(np.sum((a4 - ans_vect) * (a4 - ans_vect))))

	# backprop 
	Delta4 = a4 - ans_vect 
	Delta3 = np.dot(Delta4, Weight3) * a3 * (1 - a3) 
	Delta3 = Delta3[:, 1:] 
	Delta2 = np.dot(Delta3, Weight2) * a2 * (1 - a2) 
	Delta2 = Delta2[:, 1:] 

	# gradient 
	Weight1[:, 0] = 0
	Weight1_grad = (1 / m) * np.dot(Delta2.transpose(), a1)
	Weight2[:, 0] = 0
	Weight2_grad = (1 / m) * np.dot(Delta3.transpose(), a2)
	Weight3[:, 0] = 0
	Weight3_grad = (1 / m) * np.dot(Delta4.transpose(), a3)
	grad = np.concatenate((Weight1_grad.flatten(), Weight2_grad.flatten(), Weight3_grad.flatten())) 

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
hidden_size = 38
output_size = 10 # Números de 0 a 9

# Inicializar os pesos iniciais aleatoriamente
Weight1 = initialise(hidden_size, input_size)
Weight2 = initialise(hidden_size, hidden_size) 
Weight3 = initialise(output_size, hidden_size) 

# Unrolling parameters into a single column vector 
initial_params = np.concatenate((Weight1.flatten(), Weight2.flatten(), Weight3.flatten())) 
myargs = (input_size, hidden_size, output_size, mnist_data_train, mnist_label_train) 

# Calling minimize function to minimize cost function and to train weights 
results = minimize(neural_network, x0=initial_params, args=myargs, options={'maxiter': 100}, method="L-BFGS-B", jac=True) 

solution = results["x"] # Trained Theta is extracted 

# Weights are split back to Theta1, Theta2 
Weight1 = np.reshape(solution[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)) # shape = (38, 785) 
Weight2 = np.reshape(solution[hidden_size * (input_size + 1):hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1)], (hidden_size, hidden_size + 1)) # shape = (38, 39) 
Weight3 = np.reshape(solution[hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1):], (output_size, hidden_size + 1)) # shape = (10, 39) 

# Checking test set accuracy of our model 
pred = predict(Weight1, Weight2, Weight3, mnist_data_test) 
print('Test Set Accuracy: {:f}'.format((np.mean(pred == mnist_label_test) * 100))) 

# Checking train set accuracy of our model 
pred = predict(Weight1, Weight2, Weight3, mnist_data_train) 
print('Training Set Accuracy: {:f}'.format((np.mean(pred == mnist_label_train) * 100)))
