import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def view_mnist_data():
    print("x_train",x_train)
    print("y_train",y_train)
    print(len(x_train))
    print(len(y_train))
    print(len(y_test))


def display_number(index:int):
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"Label: {y_test[index]}")
    plt.axis('off')
    plt.show()    

   



input_size = 784      
hidden_size1 = 128   # number of nuerons in hidden layer 1 
hidden_size2 = 64    # number of nuerons in hidden layer 2
output_size = 10     # number of nuerons in the output layer(digits from 0-9) 

learning_rate = 0.01

# Rectified linear unit function(standard activation function in neural networks)
def relu(x):
    return np.maximum(0, x)

# returns 1 if x>0 or else returnz 0
def relu_derivative(x):
    return (x > 0).astype(float)

# normalises an 1d array between 0 and 1 and makes sure their sum is equal to 1(used in the last layer to fetch results) 
def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# The standard loss function 
def cross_entropy(predictions, labels):
    return -np.sum(labels * np.log(predictions + 1e-9)) / predictions.shape[0]


# Initialize weights and biases 
np.random.seed(0)  # to make sure the same random values each time we run the code
W1 = np.random.randn(input_size, hidden_size1) * 0.01
b1 = np.zeros((1, hidden_size1))

W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
b2 = np.zeros((1, hidden_size2))

W3 = np.random.randn(hidden_size2, output_size) * 0.01
b3 = np.zeros((1, output_size))


def train(n:int=1000,epochs:int=20):
    global W1, W2, W3, b1, b2, b3

    for epoch in range(epochs):
        total_loss = 0
        # first n samples
        for i in range(n):
            x = x_train[i].flatten().reshape(1, 784) / 255.0  
            y_label = y_train[i]
            y_true = np.zeros((1, 10))
            y_true[0, y_label] = 1

            # Forward
            z1 = np.dot(x, W1) + b1
            a1 = relu(z1)

            z2 = np.dot(a1, W2) + b2
            a2 = relu(z2)

            z3 = np.dot(a2, W3) + b3
            a3 = softmax(z3)

            loss = cross_entropy(a3, y_true)
            total_loss += loss

            # Backprop
            dz3 = (a3 - y_true)
            dW3 = np.dot(a2.T, dz3)
            db3 = np.sum(dz3, axis=0, keepdims=True)

            dz2 = np.dot(dz3, W3.T) * relu_derivative(z2)
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)
            dW1 = np.dot(x.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Update
            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3

            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Average Loss: {total_loss / len(x_train):.4f}")

np.savez("trained_weights.npz", 
         W1=W1, b1=b1, 
         W2=W2, b2=b2, 
         W3=W3, b3=b3)
print("✅ Weights and biases saved to trained_weights.npz")            



def load_weights(file_path: str = "trained_weights.npz"):
    global W1, W2, W3, b1, b2, b3
    data = np.load(file_path)
    W1, b1 = data["W1"], data["b1"]
    W2, b2 = data["W2"], data["b2"]
    W3, b3 = data["W3"], data["b3"]
    print("✅ Weights loaded successfully")
                 




# test the trained model
def predict(x):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)
    probabilities = np.array(a3).flatten()
    percentage_dict = {i: float(round(p * 100, 2)) for i, p in enumerate(probabilities)}

    print(percentage_dict)
    return np.argmax(a3, axis=1)  # returns the predicted digit (0-9)



# Train the model
train(n=3000,epochs=5)


# Loops through first 100 images through test data
for i in range(100):
    input = x_test[i].flatten().reshape(1, 784) / 255.0
    label = y_test[i]
    prediction = predict(input)
    print(f"Actual={label}, Predicted={prediction[0]}")



