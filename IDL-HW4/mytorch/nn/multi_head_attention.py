from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initialize your scaled dot product attention layer
        self.attention = NotImplementedError
        
        # Initialize your linear layer
        #  embed_dim -> embed_dim
        self.q_proj   = NotImplementedError
        self.k_proj   = NotImplementedError
        self.v_proj   = NotImplementedError
        self.out_proj = NotImplementedError

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        
        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project the query, key, and value inputs into query, key, and value
        # (N, L, E) -> (N, L, embed_dim)
        q = NotImplementedError
        # (N, S, E) -> (N, S, embed_dim)
        k = NotImplementedError
        # (N, S, E) -> (N, S, embed_dim)
        v = NotImplementedError

        # Split the query, key, and value into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        q = NotImplementedError
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        k = NotImplementedError
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        v = NotImplementedError

        # Merge the masks
        # (N, S) + (L, S) -> (N, H, L, S)
        mask = NotImplementedError

        # Apply the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads)
        attn_outputs = NotImplementedError

        # Merge the attention outputs   
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)
        attn_output = NotImplementedError

        # Project the attention outputs
        # (N, L, embed_dim) -> (N, L, embed_dim)
        output = NotImplementedError

        # Return output
        raise NotImplementedError

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """

        # TODO: Implement backward pass 

        # Backpropagate through the output projection   
        # (N, L, embed_dim) -> (N, L, embed_dim) 
        d_attn_output = NotImplementedError

        # Split the gradients into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        d_attn_outputs = NotImplementedError

        # Backpropagate through the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        d_q, d_k, d_v = NotImplementedError

        # Merge the gradients
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)    
        d_q = NotImplementedError
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_k = NotImplementedError
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_v = NotImplementedError

        # Backpropagate through the input projections   
        # (N, L, embed_dim) -> (N, L, embed_dim)
        d_q = NotImplementedError
        # (N, S, embed_dim) -> (N, S, embed_dim)
        d_k = NotImplementedError
        # (N, S, embed_dim) -> (N, S, embed_dim)
        d_v = NotImplementedError

        # Return gradients d_q, d_k, d_v
        raise NotImplementedError

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        # TODO: Implement merge masks

        # Expand key_padding_mask to (N, 1, 1, S) and broadcast to (N, H, L, S)
        key_mask = NotImplementedError
        
        # Expand attn_mask to (1, 1, L, S) and broadcast to (N, H, L, S)
        attention_mask = NotImplementedError
        
        # Combine masks using logical_or - if either mask is True, we want to mask that position
        combined_mask = NotImplementedError
        
        # Return combined mask
        raise NotImplementedError

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        # TODO: Implement split heads

        # Reshape: (N, L, embed_dim) -> (N, L, num_heads, embed_dim // num_heads)
        x = NotImplementedError
        
        # Transpose: (N, L, num_heads, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        x = NotImplementedError
        
        # Return x
        raise NotImplementedError

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # TODO: Implement concat heads
        # Transpose: (N, num_heads, L, embed_dim // num_heads) -> (N, L, num_heads, embed_dim // num_heads)
        x = NotImplementedError
        
        # Reshape: (N, L, num_heads, embed_dim // num_heads) -> (N, L, embed_dim)
        x = NotImplementedError
        
        # Return x
        raise NotImplementedError
