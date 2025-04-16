import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # Compute the softmax in a numerically stable way
        # Shift the input to avoid overflow by subtracting the max value along the specified dimension
        shifted_Z = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(shifted_Z)
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)
        self.A = exp_Z / sum_exp_Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            # Reshape to (batch_size, C) for easier computation
            A_reshaped = self.A.reshape(-1, C)
            dLdA_reshaped = dLdA.reshape(-1, C)
        else:
            A_reshaped = self.A
            dLdA_reshaped = dLdA

        # Compute the Jacobian of softmax
        # For each sample, we need to compute the Jacobian matrix
        batch_size = A_reshaped.shape[0]
        dLdZ_reshaped = np.zeros_like(A_reshaped)
        
        for i in range(batch_size):
            a = A_reshaped[i]
            # Compute the Jacobian matrix for this sample
            # J[i,j] = a[i] * (1[i=j] - a[j])
            J = np.diag(a) - np.outer(a, a)
            dLdZ_reshaped[i] = np.dot(J, dLdA_reshaped[i])

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            dLdZ = dLdZ_reshaped.reshape(shape)
        else:
            dLdZ = dLdZ_reshaped

        return dLdZ
 

    