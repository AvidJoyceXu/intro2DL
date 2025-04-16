import torch
import numpy as np
from torch.nn.functional import scaled_dot_product_attention as PytorchScaledDotProductAttention
from mytorch.nn.scaled_dot_product_attention import ScaledDotProductAttention as MytorchScaledDotProductAttention


def test_scaled_dot_product_attention():
    """
    Test the Scaled Dot Product Attention activation function's forward and backward passes
    """

    print("Testing Scaled Dot Product Attention ...")
    test_scaled_dot_product_attention_forward()
    test_scaled_dot_product_attention_backward()

def test_scaled_dot_product_attention_forward():
    """
    Test the Scaled Dot Product Attention activation function's forward pass
    """

    # Set the seed for reproducibility
    torch.manual_seed(11785)
    np.random.seed(11785)
    num_tests = 5

    for _ in range(num_tests):

        batch_size = np.random.choice([1, 2, 4, 8])
        num_heads  = np.random.choice([1, 2, 4, 8])
        seq_len    = np.random.choice([8, 16, 32, 64])
        embed_dim  = np.random.choice([8, 16, 32, 64])
        shape = (batch_size, num_heads, seq_len, embed_dim)

        # Create random input tensors
        query = torch.randn(shape)
        key   = torch.randn(shape)
        value = torch.randn(shape)
        
        # Ceate a mask
        mask = torch.ones(shape[:-2] + (seq_len, seq_len)).bool()
        
        # Get outputs from both implementations
        pytorch_output = PytorchScaledDotProductAttention(
            query, key, value, attn_mask=mask
        )
        
        mytorch_attention = MytorchScaledDotProductAttention()
        mytorch_output = mytorch_attention.forward(query.numpy(), key.numpy(), value.numpy(), mask=~mask.numpy())
        
        # Compare outputs
        assert  np.allclose(pytorch_output.numpy(), mytorch_output, rtol=1e-4, atol=1e-4), \
            f"Outputs don't match for case: {shape}"

    print("Test Passed: Scaled Dot Product Attention Forward")

def test_scaled_dot_product_attention_backward():
    """
    Test the Scaled Dot Product Attention activation function's backward pass
    """
    # Set the seed for reproducibility
    torch.manual_seed(11785)
    np.random.seed(11785)
    num_tests = 5

    for _ in range(num_tests):
        batch_size = np.random.choice([1, 2, 4, 8])
        num_heads  = np.random.choice([1, 2, 4, 8])
        seq_len    = np.random.choice([8, 16, 32, 64])
        embed_dim  = np.random.choice([8, 16, 32, 64])
        shape = (batch_size, num_heads, seq_len, embed_dim)


        # Create random input tensors
        query = torch.randn(shape, requires_grad=True)
        key   = torch.randn(shape, requires_grad=True)
        value = torch.randn(shape, requires_grad=True)
        
        # # Ceate a mask
        mask = torch.ones(shape[:-2] + (seq_len, seq_len)).bool()
        
        # Forward pass with PyTorch
        pytorch_output = PytorchScaledDotProductAttention(
            query, key, value, attn_mask=mask
        )
        # Backward pass with .sum()
        pytorch_output.sum().backward()
        # Get PyTorch gradients
        pytorch_dQ = query.grad.detach().numpy()
        pytorch_dK = key.grad.detach().numpy()
        pytorch_dV = value.grad.detach().numpy()
        
        # Forward pass with MyTorch
        mytorch_attention = MytorchScaledDotProductAttention()
        mytorch_output = mytorch_attention.forward(
            query.detach().numpy(), 
            key.detach().numpy(), 
            value.detach().numpy(), 
            mask=~mask.numpy()
        )
        
        # Backward pass with with np.ones_like since gradient of .sum() wrt to input is 1    
        mytorch_dQ, mytorch_dK, mytorch_dV = mytorch_attention.backward(np.ones_like(mytorch_output))
        
        # Compare gradients
        assert np.allclose(pytorch_dQ, mytorch_dQ, rtol=1e-4, atol=1e-4), \
            f"Query gradients don't match for case: {shape}"
        assert np.allclose(pytorch_dK, mytorch_dK, rtol=1e-4, atol=1e-4), \
            f"Key gradients don't match for case: {shape}"
        assert np.allclose(pytorch_dV, mytorch_dV, rtol=1e-4, atol=1e-4), \
            f"Value gradients don't match for case: {shape}"

    print("Test Passed: Scaled Dot Product Attention Backward")