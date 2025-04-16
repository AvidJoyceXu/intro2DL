import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchinfo import summary

'''
This file implements speech feature embedding for ASR (Automatic Speech Recognition) tasks.
It contains three key components:

1. StackedBLSTMEmbedding: Multi-layer bidirectional LSTM with interleaved pooling
   - Processes input sequences using two bidirectional LSTM layers
   - Reduces sequence length using max-pooling between LSTM layers
   - Handles variable-length sequences using packed sequence operations
   - Architecture flow:
     * BLSTM1 ->  MaxPool1 -> 
     * BLSTM2 ->  MaxPool2 -> 
     * Linear -> Dropout
   - Automatically calculates optimal pooling strides to achieve desired time reduction
   - Returns processed sequences with properly adjusted lengths

2. Conv2DSubsampling: Time-focused convolutional downsampling
   - Reduces sequence length using two stacked Conv2D layers
   - Maintains feature dimension while applying time-only stride
   - Uses GELU activation between convolutions
   - Automatically factors time stride into balanced conv1/conv2 strides
   - Final linear projection and optional dropout
   - Properly tracks and calculates sequence lengths through convolutions
   - Architecture flow:
     * Conv2D1 (time_stride1) -> GELU -> 
     * Conv2D2 (time_stride2) -> GELU ->
     * Linear projection -> Dropout

3. SpeechEmbedding: Flexible speech feature processor
   - Supports three reduction methods: 'conv', 'lstm', or 'both'
   - Automatically distributes time reduction between CNN and LSTM components
   - For 'both' mode, finds optimal factorization of total reduction
   - Maintains consistent output dimension throughout processing
   - Properly propagates and adjusts sequence lengths


Architecture Flow:
1. Input features (batch_size, seq_len, input_dim)
2. Conv2D downsampling (reduction based on chosen method)
3. Optional StackedBLSTM processing (for 'lstm' or 'both' methods)
4. Output features (batch_size, reduced_seq_len, output_dim) with adjusted lengths
'''

## -------------------------------------------------------------------------------------------------
## StackedBLSTMEmbedding Class
## -------------------------------------------------------------------------------------------------    
class StackedBLSTMEmbedding(nn.Module):
    """
    Stacked Bidirectional LSTM with interleaved max-pooling layers.
    Architecture: BLSTM1 -> LayerNorm1 -> MaxPool1 -> BLSTM2 -> LayerNorm2 -> MaxPool2 -> Linear -> Dropout
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 time_reduction: int = 2, dropout: float = 0.1):
        super(StackedBLSTMEmbedding, self).__init__()
        
        if not all(x > 0 for x in [input_dim, hidden_dim, output_dim, time_reduction]):
            raise ValueError("All dimension values must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
            
        # Calculate strides for the two pooling layers
        self.stride1, self.stride2 = self.closest_factors(time_reduction)
        
        # Pool configurations
        self.pool1_params = {
            "kernel_size": self.stride1,
            "stride": self.stride1,
            "padding": 0,
            "dilation": 1
        }
        self.pool2_params = {
            "kernel_size": self.stride2,
            "stride": self.stride2,
            "padding": 0,
            "dilation": 1
        }
        
        # First BLSTM layer
        self.blstm1 = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Second BLSTM layer
        self.blstm2 = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Max pooling layers
        self.pool1 = nn.MaxPool1d(**self.pool1_params)
        self.pool2 = nn.MaxPool1d(**self.pool2_params)
        
        # Final linear embedding and dropout
        self.linear_embed = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def closest_factors(self, n):
        """
        Find two closest factors of n that can be used as strides.
        Returns the larger factor first.
        """
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        return max(factor, n // factor), min(factor, n // factor)
    
    def calculate_pool_output_length(self, L_in: torch.Tensor, pool_params: dict) -> torch.Tensor:
        """
        Calculate output length for a pooling layer using the formula:
        L_out = floor((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        """
        numerator = (L_in + 2 * pool_params["padding"] - 
                    pool_params["dilation"] * (pool_params["kernel_size"] - 1) - 1)
        return (numerator // pool_params["stride"] + 1).to(torch.long)
    
    def calculate_downsampled_length(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate the downsampled length after all pooling operations.
        """
        # Apply pool1 length calculation
        lengths = self.calculate_pool_output_length(lengths, self.pool1_params)
        # Apply pool2 length calculation
        lengths = self.calculate_pool_output_length(lengths, self.pool2_params)
        return lengths

    def forward(self, x, x_len):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            x_len: Original sequence lengths (batch_size)
        Returns:
            tuple: (output tensor, downsampled lengths)
        """
        # First BLSTM
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.blstm1(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))
        
        # First max pooling
        output = output.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        output = self.pool1(output)
        output = output.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        x_len = self.calculate_pool_output_length(x_len, self.pool1_params)
        
        # Second BLSTM
        packed_input = pack_padded_sequence(output, x_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.blstm2(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=output.size(1))
        
        # Second max pooling
        output = output.transpose(1, 2)
        output = self.pool2(output)
        output = output.transpose(1, 2)
        x_len = self.calculate_pool_output_length(x_len, self.pool2_params)
        
        # Final linear embedding and dropout
        output = self.linear_embed(output)
        output = self.dropout(output)
        
        return output, x_len

## -------------------------------------------------------------------------------------------------
## Conv2DSubsampling Class
## -------------------------------------------------------------------------------------------------    
class Conv2DSubsampling(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, 
                 time_reduction: int = 2, kernel_size: int = 3):
        """
        Conv2dSubsampling module with time-only downsampling.
        
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            dropout (float): Dropout rate (default: 0.0)
            time_reduction (int): Total stride along the time dimension (default: 1)
            kernel_size (int): Size of the convolutional kernel (default: 3)
        """
        super(Conv2DSubsampling, self).__init__()
        
        self.kernel_size = kernel_size
        self.time_stride1, self.time_stride2 = self.closest_factors(time_reduction)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, kernel_size=3, stride=(self.time_stride1, 1)),
            torch.nn.GELU(),
            torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=(self.time_stride2, 1)),
            torch.nn.GELU(),
        )

        linear_in_dim = self.calculate_downsampled_length(input_dim, 1, 1)
        linear_in_dim *= output_dim
        # TODO: Replace the Linear layer with adaptive pooling and test the performance
        # self.freq_pool = torch.nn.AdaptiveAvgPool2d((None, 1))  # Pool across frequency only
        self.linear_out = torch.nn.Linear(linear_in_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, x_len):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            x_len (torch.Tensor): Non-padded lengths (batch_size)

        Returns:
            torch.Tensor: Downsampled output of shape (batch_size, new_seq_len, output_dim).
        """
        x = x.unsqueeze(1)  # Add a channel dimension for Conv2D
        x = self.conv(x)
        
        # TODO: Apply frequency pooling and reshape
        # x = self.freq_pool(x)  # (batch, channels, time, 1)
        # x = x.squeeze(-1).transpose(1, 2)  # (batch, time, channels)
        
        x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), -1) # (batch, time, channels)
        x = self.linear_out(x) # (batch, time, channels)
        x = self.dropout(x)
        
        x_len = self.calculate_downsampled_length(x_len, self.time_stride1, self.time_stride2)
        return x, x_len

    def closest_factors(self, n):
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        # Return the factor pair
        return max(factor, n // factor), min(factor, n // factor)
    
    def calculate_downsampled_length(self, lengths: torch.Tensor, stride1: int, stride2: int) -> torch.Tensor:
        """
        Calculate the downsampled length for a given sequence length and strides.
        
        Args:
            lengths (torch.Tensor): Original sequence lengths (batch_size)
            stride1 (int): Stride for first conv layer
            stride2 (int): Stride for second conv layer
            
        Returns:
            torch.Tensor: Length after downsampling (batch_size)
        """ 
        lengths = (lengths - (self.kernel_size - 1) - 1) // stride1 + 1
        lengths = (lengths - (self.kernel_size - 1) - 1) // stride2 + 1
        return lengths

## -------------------------------------------------------------------------------------------------
## SpeechEmbedding Class
## -------------------------------------------------------------------------------------------------        
class SpeechEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, time_reduction: int = 6, 
                 reduction_method: str = 'lstm', dropout: float = 0.0):
        """
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            time_reduction (int): Total time reduction factor
            reduction_method (str): Where to apply time reduction - 'conv', 'lstm', or 'both'
            dropout (float): Dropout rate
        """
        super(SpeechEmbedding, self).__init__()
        
        if not all(x > 0 for x in [input_dim, output_dim, time_reduction]):
            raise ValueError("All dimension values must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if reduction_method not in ['conv', 'lstm', 'both']:
            raise ValueError("reduction_method must be 'conv', 'lstm', or 'both'")

        self.embedding_dim = output_dim
        self.reduction_method = reduction_method

        # Calculate time reduction factors for conv and lstm
        if reduction_method == 'conv':
            conv_reduction = time_reduction
            lstm_reduction = 1
        elif reduction_method == 'lstm':
            conv_reduction = 1
            lstm_reduction = time_reduction
        else:  # 'both'
            # Split reduction between conv and lstm (try to make them similar)
            lstm_reduction, conv_reduction = self.closest_factors(time_reduction)

        # Initialize layers based on reduction method
        self.cnn = None
        self.blstm = None
        
        if reduction_method in ['conv', 'both']:
            self.cnn = Conv2DSubsampling(
                input_dim, 
                self.embedding_dim, 
                dropout=dropout,
                time_reduction=conv_reduction
            )

        if reduction_method in ['lstm', 'both']:
            lstm_input_dim = self.embedding_dim if self.cnn else input_dim
            self.blstm = StackedBLSTMEmbedding(
                input_dim=lstm_input_dim,
                hidden_dim=self.embedding_dim,
                output_dim=self.embedding_dim,
                time_reduction=lstm_reduction,
                dropout=dropout
            )

    def closest_factors(self, n):
        """Find two closest factors of n."""
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        return max(factor, n // factor), min(factor, n // factor)

    def forward(self, x, x_len):
        """
        Args:
            x     : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            tuple: (output tensor, downsampled lengths)
        """
        if self.cnn is not None:
            x, x_len = self.cnn(x, x_len)
        if self.blstm is not None:
            x, x_len = self.blstm(x, x_len)
        return x, x_len
    
    def calculate_downsampled_length(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate the downsampled length for a given sequence length.
        """
        if self.cnn is not None:
            lengths = self.cnn.calculate_downsampled_length(lengths, self.cnn.time_stride1, self.cnn.time_stride2)
        if self.blstm is not None:
            lengths = self.blstm.calculate_downsampled_length(lengths)
        return lengths


## -------------------------------------------------------------------------------------------------
## Test Cases
## -------------------------------------------------------------------------------------------------

def get_inputs(input_dim: int, max_length: int, batch_size: int):
    input_tensor = torch.randn(batch_size, max_length, input_dim)
    input_lengths = torch.ones(batch_size) * max_length
    return input_tensor, input_lengths

def test_speech_embedding_lstm(time_reduction: int = 2):
    input_dim  = 80
    output_dim = 256
    max_length = 1000
    batch_size = 10
    input_tensor, input_lengths = get_inputs(input_dim, max_length, batch_size)
    model = SpeechEmbedding(input_dim, output_dim, time_reduction=time_reduction, reduction_method='lstm', dropout=0.1)
    summary(model, input_data=[input_tensor, input_lengths])
    output, output_lengths = model(input_tensor, input_lengths)
    print(output_lengths)

def test_speech_embedding_conv(time_reduction: int = 2):
    input_dim  = 80
    output_dim = 256
    max_length = 1000
    batch_size = 10
    input_tensor, input_lengths = get_inputs(input_dim, max_length, batch_size)
    model = SpeechEmbedding(input_dim, output_dim, time_reduction=time_reduction, reduction_method='conv', dropout=0.1)
    summary(model, input_data=[input_tensor, input_lengths])
    output, output_lengths = model(input_tensor, input_lengths)
    print(output_lengths)

def test_speech_embedding_both(time_reduction: int = 2):
    input_dim  = 80
    output_dim = 256
    max_length = 1000
    batch_size = 10
    input_tensor, input_lengths = get_inputs(input_dim, max_length, batch_size)
    model = SpeechEmbedding(input_dim, output_dim, time_reduction=time_reduction, reduction_method='both', dropout=0.1)
    summary(model, input_data=[input_tensor, input_lengths])
    output, output_lengths = model(input_tensor, input_lengths)
    print(output_lengths)

if __name__ == "__main__":
    #test_speech_embedding_lstm(time_reduction=2)
    #test_speech_embedding_conv(time_reduction=2)
    #test_speech_embedding_both(time_reduction=2)
    #test_speech_embedding_lstm(time_reduction=3)
    #test_speech_embedding_conv(time_reduction=3)
    #test_speech_embedding_both(time_reduction=3)
    #test_speech_embedding_lstm(time_reduction=4)
    #test_speech_embedding_conv(time_reduction=4)
    #test_speech_embedding_both(time_reduction=4)
    pass
