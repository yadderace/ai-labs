
import numpy as np

class Loss:
    def calculate(self, y, y_hat):
        sample_losses = self.forward(y, y_hat)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalEntropy(Loss):
    def forward(self, y, y_hat):
        samples = len(y)
        # This is to avoid log(0)
        y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)

        correct_confidences = np.sum(y_hat_clipped * y, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

        

class MyNeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        
        # Generating the weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        # Generating the biases
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.bias_hidden_output = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def forward(self, X):
        
        # Calculating the hidden layer
        self.hidden_layer = self.relu(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        # Calculating the output layer
        self.output_layer = self.softmax(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_hidden_output)
        
        return self.output_layer

    def backward(self, X, y, y_hat, learning_rate = 0.01):
        m = X.shape[0]

        # Output layer error
        output_error = y_hat - y
        d_weights2 = np.dot(self.hidden_layer.T, output_error) / m
        d_bias2 = np.sum(output_error, axis=0, keepdims=True) / m
        
        # Hidden layer error
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_layer)
        d_weights1 = np.dot(X.T, hidden_error) / m
        d_bias1 = np.sum(hidden_error, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.weights_hidden_output -= learning_rate * d_weights2
        self.bias_hidden_output -= learning_rate * d_bias2
        self.weights_input_hidden -= learning_rate * d_weights1
        self.bias_input_hidden -= learning_rate * d_bias1


    def train(self, X, y, epochs=1000, learning_rate=0.05):
        
        for epoch in range(epochs):
            y_hat = self.forward(X)
            self.backward(X, y, y_hat, learning_rate)

            if epoch % 100 == 0:
                loss_function = LossCategoricalEntropy()
                loss = loss_function.calculate(y, y_hat)
                print(f"Loss: {loss}")

                # Calculating the accuracy
                predictions = np.argmax(y_hat, axis=1)
                accuracy = np.mean(predictions == np.argmax(y, axis=1))
                print(f"Accuracy: {accuracy}")

        


        
     

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
    nn.train(X, y)
    

if __name__ == "__main__":
    main()