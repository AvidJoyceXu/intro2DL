import torch
import numpy as np
from mytorch.nn.multi_head_attention import MultiHeadAttention as MytorchMultiHeadAttention
from torch.nn import MultiheadAttention as PytorchMultiHeadAttention


def test_multi_head_attention():
    """
    Test the Multi Head Attention activation function's forward and backward passes
    """

    print("Testing Multi Head Attention ...")
    test_multi_head_attention_forward()
    test_multi_head_attention_backward()


def test_multi_head_attention_forward():
    """
    Test the Multi Head Attention activation function's forward pass
    """
    # Set the seed for reproducibility
    torch.manual_seed(11785)
    np.random.seed(11785)
    num_tests = 5

    for _ in range(num_tests):
        # Random dimensions
        batch_size = np.random.choice([1, 2, 4, 8])
        seq_len    = np.random.choice([8, 16, 32, 64])
        embed_dim  = np.random.choice([8, 16, 32, 64])
        num_heads  = np.random.choice([2, 4, 8])  # ensure embed_dim is divisible by num_heads
        
        # Initialize both implementations
        mytorch_mha = MytorchMultiHeadAttention(embed_dim, num_heads)
        pytorch_mha = PytorchMultiHeadAttention(embed_dim, num_heads, bias=True, batch_first=True)
        
        # Copy weights from PyTorch to MyTorch implementation
        mytorch_mha.init_weights(
            pytorch_mha.in_proj_weight[:embed_dim, :].detach().numpy(),
            pytorch_mha.in_proj_bias[:embed_dim].detach().numpy(),
            pytorch_mha.in_proj_weight[embed_dim:2*embed_dim, :].detach().numpy(),
            pytorch_mha.in_proj_bias[embed_dim:2*embed_dim].detach().numpy(),
            pytorch_mha.in_proj_weight[2*embed_dim:, :].detach().numpy(),
            pytorch_mha.in_proj_bias[2*embed_dim:].detach().numpy(),
            pytorch_mha.out_proj.weight.detach().numpy(),
            pytorch_mha.out_proj.bias.detach().numpy()   
        )
        
        # Create input tensors
        query = torch.randn(batch_size, seq_len, embed_dim)
        key   = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create key padding mask (attention to all positions)
        key_padding_mask = torch.zeros(batch_size, seq_len).bool()

        # Create attention mask (attention to all positions)    
        attn_mask = torch.zeros(seq_len, seq_len).bool()

        # Get outputs from both implementations
        pytorch_output, _ = pytorch_mha(
            query, 
            key, 
            value, 
            key_padding_mask=key_padding_mask, 
            attn_mask=attn_mask,
            need_weights=False,
        )

        mytorch_output = mytorch_mha.forward(
            query.numpy(), 
            key.numpy(), 
            value.numpy(), 
            key_padding_mask=key_padding_mask.numpy(), 
            attn_mask=attn_mask.numpy()
        )

        # Compare outputs
        assert np.allclose(pytorch_output.detach().numpy(), mytorch_output, rtol=1e-4, atol=1e-4), \
            f"Outputs don't match for case: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}"

    print("Test Passed: Multi Head Attention Forward")


def test_multi_head_attention_backward():
    """
    Test the Multi Head Attention activation function's backward pass
    """
    # Set the seed for reproducibility
    torch.manual_seed(11785)
    np.random.seed(11785)
    num_tests = 5

    for _ in range(num_tests):
        # Random dimensions
        batch_size = np.random.choice([1, 2, 4, 8])
        seq_len    = np.random.choice([8, 16, 32, 64])
        embed_dim  = np.random.choice([8, 16, 32, 64])
        num_heads  = np.random.choice([2, 4, 8])

        # Initialize both implementations
        mytorch_mha = MytorchMultiHeadAttention(embed_dim, num_heads)
        pytorch_mha = PytorchMultiHeadAttention(embed_dim, num_heads, bias=True, batch_first=True)
        
        # Copy weights from PyTorch to MyTorch implementation
        mytorch_mha.init_weights(
            pytorch_mha.in_proj_weight[:embed_dim, :].detach().numpy(),
            pytorch_mha.in_proj_bias[:embed_dim].detach().numpy(),
            pytorch_mha.in_proj_weight[embed_dim:2*embed_dim, :].detach().numpy(),
            pytorch_mha.in_proj_bias[embed_dim:2*embed_dim].detach().numpy(),
            pytorch_mha.in_proj_weight[2*embed_dim:, :].detach().numpy(),
            pytorch_mha.in_proj_bias[2*embed_dim:].detach().numpy(),
            pytorch_mha.out_proj.weight.detach().numpy(),
            pytorch_mha.out_proj.bias.detach().numpy()   
        )

        # Create input tensors with gradients
        query = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        key   = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        value = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        
        # Create masks
        key_padding_mask = torch.zeros(batch_size, seq_len).bool()
        attn_mask = torch.zeros(seq_len, seq_len).bool()

        # Forward and backward pass with PyTorch
        pytorch_output, _ = pytorch_mha(
            query, key, value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )
        pytorch_output.sum().backward()
        
        # Get PyTorch gradients
        pytorch_dQ = query.grad.detach().numpy()
        pytorch_dK = key.grad.detach().numpy()
        pytorch_dV = value.grad.detach().numpy()
        
        # Forward and backward pass with MyTorch
        mytorch_output = mytorch_mha.forward(
            query.detach().numpy(),
            key.detach().numpy(),
            value.detach().numpy(),
            key_padding_mask=key_padding_mask.numpy(),
            attn_mask=attn_mask.numpy()
        )
        
        # Backward pass with ones since gradient of sum() is 1
        mytorch_dQ, mytorch_dK, mytorch_dV = mytorch_mha.backward(np.ones_like(mytorch_output))

        # Compare gradients
        assert np.allclose(pytorch_dQ, mytorch_dQ, rtol=1e-4, atol=1e-4), \
            f"Query gradients don't match for case: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}"
        assert np.allclose(pytorch_dK, mytorch_dK, rtol=1e-4, atol=1e-4), \
            f"Key gradients don't match for case: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}"
        assert np.allclose(pytorch_dV, mytorch_dV, rtol=1e-4, atol=1e-4), \
            f"Value gradients don't match for case: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}"

    print("Test Passed: Multi Head Attention Backward")    