import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

'''
TODO: Implement these Modules.

This file contains two key decoder layer implementations used in transformer architectures:

1. SelfAttentionDecoderLayer: Used in decoder-only transformers (like GPT)
   - Contains masked self-attention and feed-forward sublayers
   - Used for tasks like language modeling where only previous tokens can be attended to
   
2. CrossAttentionDecoderLayer: Used in encoder-decoder transformers (like BART)
   - Contains masked self-attention, cross-attention, and feed-forward sublayers
   - Used for tasks like translation where decoder needs to attend to encoder outputs

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps for Each Layer:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for masked self-attention
   - CrossAttentionLayer for cross-attention (in CrossAttentionDecoderLayer only)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate masks to attention layers
   - Return both outputs and attention weights
'''

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.  
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        ''' 
        super().__init__()
        # TODO: Implement __init__
       
        # TODO: Initialize the sublayers      
        self.self_attn = NotImplementedError # Masked self-attention layer
        self.ffn = NotImplementedError # Feed-forward network
        raise NotImplementedError # Remove once implemented

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the DecoderLayer1.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (torch.Tensor): The padding mask for the decoder. shape: (batch_size, seq_len)
            attn_mask (torch.Tensor): The self-attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, num_classes)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        # TODO: Implement forward: Follow the figure in the writeup

        x, mha_attn_weights = NotImplementedError, NotImplementedError
        
        # TODO: Return the output tensor and attention weights
        raise NotImplementedError # Remove once implemented

## -------------------------------------------------------------------------------------------------    
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Used in the encoder-decoder Transformer architecture.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the CrossAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # TODO: Implement __init__

        # TODO: Initialize the sublayers  
        self.self_attn  = NotImplementedError # Masked self-attention layer
        self.cross_attn = NotImplementedError # Cross-attention layer
        self.ffn        = NotImplementedError # Feed-forward network
        raise NotImplementedError # Remove once implemented

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, dec_key_padding_mask: Optional[torch.Tensor] = None, enc_key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionDecoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            enc_output (torch.Tensor): The encoder output. shape: (batch_size, seq_len, d_model)
            dec_key_padding_mask (Optional[torch.Tensor]): The padding mask for the decoder input. shape: (batch_size, seq_len)
            enc_key_padding_mask (Optional[torch.Tensor]): The padding mask for the encoder output. shape: (batch_size, seq_len')
            attn_mask (Optional[torch.Tensor]): The self-attention mask for the decoder input. shape: (seq_len, seq_len)
        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            self_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
            cross_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)    
        '''
        # TODO: Implement forward: Follow the figure in the writeup

        x, self_attn_weights  = NotImplementedError, NotImplementedError
        x, cross_attn_weights = NotImplementedError, NotImplementedError

        # TODO: Return the output tensor and attention weights    
        raise NotImplementedError # Remove once implemented
## -------------------------------------------------------------------------------------------------    
