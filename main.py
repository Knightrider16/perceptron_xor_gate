import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1
epochs = 10000
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 0])
weights_hidden = np.random.rand(2, 2)
bias_hidden = np.random.rand(2)
weights_output = np.random.rand(2)
bias_output = np.random.rand()

def activation(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

error_history = []
for epoch in range(epochs):
    errors = 0
    for inputs, label in zip(training_inputs, labels):
        hidden_input = np.dot(inputs, weights_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, weights_output) + bias_output
        prediction = sigmoid(final_input)
        error = label - prediction
        errors += abs(error)
        d_pred = error * sigmoid_derivative(prediction)
        weights_output += learning_rate * d_pred * hidden_output
        bias_output += learning_rate * d_pred
        d_hidden = d_pred * weights_output * sigmoid_derivative(hidden_output)
        weights_hidden += learning_rate * np.outer(inputs, d_hidden)
        bias_hidden += learning_rate * d_hidden

    error_history.append(errors)

print("Testing XOR gate after training:")
for x in training_inputs:
    hidden_input = np.dot(x, weights_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_output) + bias_output
    output = activation(final_input)
    print(f"Input: {x}, Output: {output}")

plt.plot(range(epochs), error_history, color='blue')
plt.title('Training Error over Epochs for XOR')
plt.xlabel('Epoch')
plt.ylabel('Total Error')
plt.grid()
plt.show()
