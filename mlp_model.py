import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layer_sizes):
        
        self.layer_sizes = layer_sizes
        self.parameters = {}
        self.num_layers = len(layer_sizes) - 1
        
        
        for i in range(1, len(layer_sizes)):
            
            self.parameters[f"W{i}"] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2. / layer_sizes[i-1])
            self.parameters[f"b{i}"] = np.zeros((layer_sizes[i], 1))
    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)    
    def forward(self, X):
        
        #forward propagation
        cache = {"A0": X}
        
        # hidden layers
        for i in range(1, self.num_layers):
            W = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]
            A_prev = cache[f"A{i-1}"]
            
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)
            
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = A
            
        # output layer
        W_last = self.parameters[f"W{self.num_layers}"]
        b_last = self.parameters[f"b{self.num_layers}"]
        A_prev = cache[f"A{self.num_layers-1}"]
        
        Z_last = np.dot(W_last, A_prev) + b_last
        A_last = self.softmax(Z_last)
        
        cache[f"Z{self.num_layers}"] = Z_last
        cache[f"A{self.num_layers}"] = A_last
        
        return A_last, cache      
    def compute_loss(self, Y, Y_hat):
        
        #Sum of negative log-likelihood loss.
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-9)) #for log(0) error
        return loss / m
    def backward(self, Y, cache):
        #Implement back-propagation.
        grads = {}
        L = self.num_layers
        m = Y.shape[1]
        
        # Softmax + NLL 
        dZ = cache[f"A{L}"] - Y
        
        grads[f"dW{L}"] = (1/m) * np.dot(dZ, cache[f"A{L-1}"].T)
        grads[f"db{L}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        for i in range(L-1, 0, -1):
            dZ_next = dZ
            W_next = self.parameters[f"W{i+1}"]
            
            # ReLU derivation
            dA = np.dot(W_next.T, dZ_next)
            dZ = dA * (cache[f"Z{i}"] > 0) # Z > 0 ise 1, değilse 0
            
            grads[f"dW{i}"] = (1/m) * np.dot(dZ, cache[f"A{i-1}"].T)
            grads[f"db{i}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
        return grads
    def update_parameters(self, grads, lr):
        #Update parameters w and b using gradient descent.
        for i in range(1, self.num_layers + 1):
            self.parameters[f"W{i}"] -= lr * grads[f"dW{i}"]
            self.parameters[f"b{i}"] -= lr * grads[f"db{i}"]
    def gradient_check(self, X, Y, epsilon=1e-7):
        #Implement numerical approximation of gradients 
        print("Performing Gradient Check (Checking W1[0,0])...")
        
        # 1. Analytic Gradient
        _, cache = self.forward(X)
        grads = self.backward(Y, cache)
        grad_analytic = grads["dW1"][0, 0]
        
        # 2. numerical gradient
        original_W = self.parameters["W1"][0, 0]
        
        # J_plus
        self.parameters["W1"][0, 0] = original_W + epsilon
        A_plus, _ = self.forward(X)
        loss_plus = self.compute_loss(Y, A_plus)
        
        # J_minus
        self.parameters["W1"][0, 0] = original_W - epsilon
        A_minus, _ = self.forward(X)
        loss_minus = self.compute_loss(Y, A_minus)
        
        # Reset
        self.parameters["W1"][0, 0] = original_W
        
        grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)
        
        diff = abs(grad_analytic - grad_numeric) / (abs(grad_numeric) + abs(grad_analytic) + 1e-9)
        print(f"Analytic: {grad_analytic}, Numeric: {grad_numeric}, Diff: {diff}")
        
        if diff < 1e-5:
            print("Gradient Check Passed!")
        else:
            print("Gradient Check Warning!")
    def visualize_weights(self):
        #Visualize learned parameters as images
        W1 = self.parameters["W1"]
        # first 5 neurons
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            # weight to 128x128 image (input size)
            dim = int(np.sqrt(W1.shape[1]))
            try:
                img = W1[i, :].reshape(dim, dim)
                axes[i].imshow(img, cmap='viridis')
                axes[i].axis('off')
                axes[i].set_title(f"Neuron {i+1}")
            except: pass
        plt.show()

def train_mlp(X_train, Y_train, X_test, Y_test, layer_sizes, epochs, lr, batch_size):
    
    model = MLP(layer_sizes)
    
    # Transpose for (Features, Samples) format
    X_train_T, Y_train_T = X_train.T, Y_train.T
    X_test_T, Y_test_T = X_test.T, Y_test.T
    
    m = X_train_T.shape[1]
    loss_history = []
    acc_history = []
    
    # Run Gradient Check once at start
    if m > 5:
        sample_X = X_train_T[:, :5]
        sample_Y = Y_train_T[:, :5]
        model.gradient_check(sample_X, sample_Y)
    
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(m)
        X_shuffled = X_train_T[:, perm]
        Y_shuffled = Y_train_T[:, perm]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]
            
            if X_batch.shape[1] < batch_size: continue
            
            # Forward
            A, cache = model.forward(X_batch)
            loss = model.compute_loss(Y_batch, A)
            epoch_loss += loss
            
            # Backward
            grads = model.backward(Y_batch, cache)
            model.update_parameters(grads, lr)
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Calculate Accuracy
        A_test, _ = model.forward(X_test_T)
        preds = np.argmax(A_test, axis=0)
        true_labels = np.argmax(Y_test_T, axis=0)
        acc = np.mean(preds == true_labels)
        acc_history.append(acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Acc: {acc:.4f}")
            
    return model, loss_history, acc_history