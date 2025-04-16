import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(-1)  # Apply softmax along the last dimension (S)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # Store inputs for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        # Get the embedding dimension (E) from Q
        d_k = Q.shape[-1]
        K_transposed = np.swapaxes(K, -1, -2)
        # Compute Q @ K^T and scale by sqrt(d_k)
        # Fix: Use proper transpose with axis indices instead of ellipsis
        scaled_dot_product = np.matmul(Q, K_transposed) / np.sqrt(d_k)
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            # Convert mask to float and multiply by -self.eps
            # This will make masked positions have a very negative value
            scaled_dot_product = scaled_dot_product + (mask.astype(np.float32) * -self.eps)

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = np.matmul(self.attention_scores, V)

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions 
        # NOTE: 
        # print(self.attention_scores.shape) # (N, ..., H, L, S)
        # print(d_output.shape) # (N, ..., H, L, Ev)
        # print(self.V.shape) # (N, ..., H, S, Ev)
        d_V = np.matmul(np.swapaxes(self.attention_scores, -1, -2), d_output) 
        # NOTE: (N, ..., H, S, L) @ (N, ..., H, L, Ev) -> (N, ..., H, S, Ev)
        
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        d_attention_scores = np.matmul(d_output, np.swapaxes(self.V, -1, -2))
        
        # Apply softmax gradient
        d_scaled_dot_product = self.softmax.backward(d_attention_scores) # (N, ..., H, L, S)
        
        # Scale gradients by sqrt(d_k)
        d_k = self.Q.shape[-1]
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(d_k)
        
        # Calculate gradients for Q and K
        # NOTE: Q - shape (N, ..., H, L, E)
        #       K - shape (N, ..., H, S, E)
        #       d_scaled_dot_product - shape (N, ..., H, L, S)
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = np.matmul(d_scaled_dot_product, self.K)
        # (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -1, -2), self.Q)
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

