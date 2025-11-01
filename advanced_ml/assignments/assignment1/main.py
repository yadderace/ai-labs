
import numpy as np
import csv

class MyNeuralNetwork:

    def __init__(
        self, 
        input_size = 8, 
        hidden_size = 3, 
        output_size = 8, 
        loss_function = 'cross_entropy',
        output_file = 'statistics.csv'
    ):
        
        # Generating the weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        # Generating the biases
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.bias_hidden_output = np.zeros((1, output_size))

        # Defining the output file for the statistics
        self.output_file = output_file

        # Initialize log file
        self.initialize_log_file()

    def initialize_log_file(self):
        # Create headers for the CSV file
        headers = ['epoch', 'loss', 'accuracy']
        
        # Add columns for each hidden neuron's Z and activation
        for i in range(self.weights_input_hidden.shape[1]):  # For each hidden neuron
            headers.extend([f'z_hidden_{i}', f'a_hidden_{i}'])
        
        # Write headers to file
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def log_training_step(self, epoch, loss, accuracy, hidden_z, hidden_activations):
        # Prepare row data
        row = [epoch, loss, accuracy]
        # Add Z and activation for each hidden neuron
        for z, a in zip(hidden_z[0], hidden_activations[0]):
            row.extend([z, a])
        
        # Append to CSV
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def forward(self, X):
        
        # Calculating the hidden layer
        # Store Z values
        self.z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        # Calculate activation
        self.hidden_layer = self.sigmoid(self.z_hidden)

        # Calculating the output layer
        self.z_output = np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_hidden_output
        self.output_layer = self.sigmoid(self.z_output)
        
        return self.output_layer, self.z_hidden

    def backward(self, X, y, y_hat, learning_rate = 0.01):
        m = X.shape[0]

        # ================================= [Output layer error] =================================
        # For loss function we use MSE
        # The loss function derivative with respect to the predicted values y_hat
        l_derivative = y_hat - y
        # The derivative of the predicted values with respect to z (W * X + b)
        # is the derivative of sigmoid
        sigmoid_derivative = self.sigmoid_derivative(y_hat) 
        # The derivative of the loss function with respect to z 
        # is the product of the loss function derivative and the sigmoid derivative
        # This is the error term
        error_term_output = l_derivative * sigmoid_derivative

        # The derivative of the loss function with respect to weights
        # is the product of the error term and the hidden layer
        # and then we divide by the number of samples
        dW_hidden_output = np.dot(self.hidden_layer.T, error_term_output) / m
        # The derivative of the loss function with respect to bias
        # is the sum of the error term
        # and then we divide by the number of samples
        db_hidden_output = np.sum(error_term_output, axis=0, keepdims=True) / m

        # ================================= [Hidden layer error] =================================
        # The derivative of the loss function with respect to weights
        
        error_term_hidden = np.dot(error_term_output, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_layer)

        # The derivative of the loss function with respect to weights
        dW_input_hidden = np.dot(X.T, error_term_hidden) / m
        # The derivative of the loss function with respect to bias
        db_input_hidden = np.sum(error_term_hidden, axis=0, keepdims=True) / m

        # ================================= [Update weights and biases] ==========================
        # Update weights and biases
        self.weights_hidden_output -= learning_rate * dW_hidden_output
        self.bias_hidden_output -= learning_rate * db_hidden_output

        self.weights_input_hidden -= learning_rate * dW_input_hidden
        self.bias_input_hidden -= learning_rate * db_input_hidden
        
    def train(self, X, y, epochs=10000, learning_rate=0.01):
        
        for epoch in range(epochs):
            # Forward pass
            y_hat, z_hidden = self.forward(X)

            self.backward(X, y, y_hat, learning_rate)

            if epoch % 100 == 0:
                # Calculate metrics
                loss = np.sum((y_hat - y) ** 2) / (2 * X.shape[0])
                predictions = np.argmax(y_hat, axis=1)
                accuracy = np.mean(predictions == np.argmax(y, axis=1))
                
                # Log to console
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Log to CSV
            self.log_training_step(
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                hidden_z=z_hidden,
                hidden_activations=self.hidden_layer
            )
                
     

def main():
    input_size = 8
    hidden_size = 3
    output_size = 8

    X = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
    ])
    y = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
    ])

    nn = MyNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    nn.train(X, y, epochs=10000, learning_rate=0.2)
    

if __name__ == "__main__":
    main()