import torch.nn as nn
import torch

def test_sublayer_selfattention(self_attn):
    '''
    Test the self-attention sublayer.
    Args:
        self_attn (nn.Module): The self-attention sublayer.
    '''
    # Structural Test
    test_initialization(self_attn)  

    # Functional Tests
    test_forward_shapes(self_attn)

    # Behavioural Test
    test_padding_mask_behaviour(self_attn)
    test_self_attention_mask_behaviour(self_attn)
    test_self_attention_residual(self_attn)
    
    

def test_initialization(self_attn):
    '''
    Test if the layers exist in the sublayer.
    Args:
        self_attn (nn.Module): The self-attention sublayer.
    '''
    print("Testing initialization ...")
    d_model     = 10    
    num_heads   = 2
    dropout     = 0.0
    model       = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout)

    # Check if the layers exist in the model 
    expected_attributes = {"mha", "norm", "dropout"}
    assert expected_attributes.issubset(dir(model)), "Required attributes are missing"
   
    # Check if the layers are not None
    assert model.mha is not None, "Multi-Head Attention layer is None"
    assert model.norm is not None, "Normalization layer is None"
    assert model.dropout is not None, "Dropout layer is None"   
   
    # Check if the layers are of the correct type   
    assert isinstance(model.mha, nn.MultiheadAttention), "Multi-Head Attention layer is not of the correct type"
    assert model.mha.embed_dim == d_model, f"Multi-Head Attention embed_dim: expected {d_model} but got {model.mha.embed_dim}"
    assert model.mha.num_heads == num_heads, f"Multi-Head Attention num_heads: expected {num_heads} but got {model.mha.num_heads}" 
    assert isinstance(model.norm, nn.LayerNorm), "Normalization layer is not of the correct type"
    assert model.norm.normalized_shape == (d_model,), f"Normalization layer normalized_shape: expected {d_model} but got {model.norm.normalized_shape}"  
    assert isinstance(model.dropout, nn.Dropout), "Dropout layer is not of the correct type"
    assert model.dropout.p == dropout, f"Dropout layer p: expected {dropout} but got {model.dropout.p}"

    print("Test Passed: All layers exist and are instantiated correctly")


def test_forward_shapes(self_attn):
    '''
    Test if the forward pass returns the correct shapes.
    Args:
        self_attn (nn.Module): The self-attention sublayer.
        pad_mask_fn (Callable[[torch.Tensor], torch.Tensor]): The padding mask function.
        attn_mask_fn (Callable[[torch.Tensor], torch.Tensor]): The attention mask function.
    '''
    print("Testing forward shapes ...")
    # Create an instance    
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout)

    # Create a random input tensor
    batch_size = 4
    seq_length = 8
    input_tensor  = torch.randn(batch_size, seq_length, d_model)
    input_lengths = torch.randint(1, seq_length, (batch_size,)) 

    # Create a random padding mask tensor
    pad_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

    # Create a random self-attention mask tensor
    attn_mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)

    # Forward pass
    output, attn_weights = model.forward(input_tensor, pad_mask, attn_mask)

    assert output.shape == (batch_size, seq_length, d_model), f"Output shape: expected {(batch_size, seq_length, d_model)} but got {output.shape}"  
    assert attn_weights.shape == (batch_size, seq_length, seq_length), f"Attention weights shape: expected {(batch_size, seq_length, seq_length)} but got {attn_weights.shape}"
    print("Test Passed: Forward pass returns the correct shapes")


def test_padding_mask_behaviour(self_attn):
    '''
    Test if the padding mask is applied correctly.
    Args:
        self_attn (nn.Module): The self-attention sublayer.
        pad_mask_fn (Callable[[torch.Tensor], torch.Tensor]): The padding mask function.
        attn_mask_fn (Callable[[torch.Tensor], torch.Tensor]): The attention mask function.
    '''
    print("Testing padding mask behaviour ...")

    print("Testing padding mask behaviour ...")
    # Create an instance
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout)
    
    # Create a random input tensor  
    batch_size = 4
    seq_length = 8
    to_pad = 2
    input_tensor = torch.randn(batch_size, seq_length, d_model)

    # Create a padding mask tensor with the last 6 tokens masked    
    pad_mask = torch.zeros(batch_size, seq_length)
    pad_mask[:, to_pad:] = 1  # Mask last 6 tokens
    pad_mask  = pad_mask.to(torch.bool)

    # Create a self-attention mask tensor of zeros  
    attn_mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)

    # Forward pass  
    _, attn_weights = model.forward(input_tensor, pad_mask, attn_mask)

    assert torch.all(attn_weights[:, :, to_pad:] == 0), "Attention weights for padded positions should be zero"
    print("Test Passed: Padding mask is applied correctly") 
    

def test_self_attention_mask_behaviour(self_attn):
    '''
    Test if the self-attention mask is applied correctly.
    '''
    print("Testing self-attention mask behaviour ...")
    # Create an instance
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout)
    
    # Create a random input tensor  
    batch_size = 4
    seq_length = 8
    input_tensor = torch.randn(batch_size, seq_length, d_model)

    # Create a padding mask tensor of zeros
    pad_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

    # Create a self-attention mask 
    attn_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()  # Upper triangular causal mask

    _, attn_weights = model.forward(input_tensor, pad_mask, attn_mask)

    # Check if the future positions are not attended
    assert torch.all(attn_weights.triu(diagonal=1) == 0), "Future positions should not be attended"
    print("Test Passed: Self-attention mask is applied correctly")  


def test_self_attention_residual(self_attn):
    '''
    Test if the self-attention residual is applied correctly.
    '''
    print("Testing self-attention residual ...")
    # Create an instance with no dropout for deterministic behavior
    d_model   = 4
    num_heads = 2
    dropout   = 0.0  # Set to 0 for deterministic behavior
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout)
    
    # Force MHA to be an identity transformation
    # By setting the projection matrices to identity
    with torch.no_grad():
        model.mha.in_proj_weight.data = torch.eye(3 * d_model, d_model)
        model.mha.in_proj_bias.data.zero_()
        model.mha.out_proj.weight.data = torch.eye(d_model)
        model.mha.out_proj.bias.data.zero_()
    
    # Create a random input tensor
    batch_size   = 4
    seq_length   = 10
    input_tensor = torch.randn(batch_size, seq_length, d_model)
    
    # Create masks (all False to allow attention)
    pad_mask  = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    attn_mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)

    # Get the output
    output, _ = model.forward(input_tensor, pad_mask, attn_mask)

    assert torch.allclose(output, input_tensor, rtol=1e-5, atol=1e-5), "Residual connection is not applied correctly."
    
    print("Test Passed: Residual connection is applied correctly")

def main():
    from hw4lib.model import SelfAttentionLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'SelfAttentionLayer': [
                {
                    'func': lambda: test_sublayer_selfattention(SelfAttentionLayer),
                    'description': 'Test the self-attention sublayer'
                }
            ]
        }
    )   

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()  
