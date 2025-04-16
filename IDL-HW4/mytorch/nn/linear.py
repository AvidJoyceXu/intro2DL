import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A

        A_reshaped = A.reshape(-1, A.shape[-1]) # (-1, in_features)
        
        # Compute Z = WA + b
        Z = np.dot(A_reshaped, self.W.T) + self.b
        
        return Z.reshape(A.shape[:-1] + (self.b.shape[0],))

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Store original shapes for reshaping back later
        original_shape_A = self.A.shape
        
        # Reshape for matrix multiplication
        A_reshaped = self.A.reshape(-1, self.A.shape[-1]) # (-1, in_features)
        dLdZ_reshaped = dLdZ.reshape(-1, dLdZ.shape[-1]) # (-1, out_features)
        
        # Compute gradients
        # dL/dA = dL/dZ * W^T
        self.dLdA = np.dot(dLdZ_reshaped, self.W).reshape(original_shape_A) # (*, in_features)
        
        # dL/dW = (dL/dZ)^T * A
        self.dLdW = np.dot(dLdZ_reshaped.T, A_reshaped) # (out_features, in_features)
        
        # dL/db = sum(dL/dZ)
        self.dLdb = np.sum(dLdZ_reshaped, axis=0) # (out_features,)
        
        # Return gradient of loss wrt input
        return self.dLdA