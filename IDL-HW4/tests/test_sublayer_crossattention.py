import torch
import torch.nn as nn

def test_sublayer_crossattention(cross_attention):
    '''
    Test the cross-attention sublayer.
    '''
    # Structural Test
    test_initialization(cross_attention)

    # Functional Tests
    test_forward_shapes(cross_attention)

    # Behavioural Tests
    test_padding_mask_behaviour(cross_attention)
    test_cross_attention_behaviour(cross_attention)
    test_cross_attention_residual(cross_attention)


def test_initialization(cross_attention):
    '''
    Test if the layers exist in the decoder sublayer 2.
    '''
    print("Testing initialization ...")
    d_model = 10
    num_heads = 2
    dropout = 0.1
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout)

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


def test_forward_shapes(cross_attention):
    '''
    Test if the forward pass returns the correct shapes.
    '''
    print("Testing forward shapes ...")

    # Create an instance
    d_model = 10
    num_heads = 2
    dropout = 0.1
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout)

    # Create random input tensors
    batch_size = 4
    dec_seq_length = 8
    enc_seq_length = 6
    decoder_input = torch.randn(batch_size, dec_seq_length, d_model)
    encoder_output = torch.randn(batch_size, enc_seq_length, d_model)

    # Create padding mask for encoder
    pad_mask_enc = torch.zeros(batch_size, enc_seq_length)

    # Forward pass
    output, attn_weights = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    assert output.shape == (batch_size, dec_seq_length, d_model), f"Output shape: expected {(batch_size, dec_seq_length, d_model)} but got {output.shape}"
    assert attn_weights.shape == (batch_size, dec_seq_length, enc_seq_length), f"Attention weights shape: expected {(batch_size, dec_seq_length, enc_seq_length)} but got {attn_weights.shape}"
    print("Test Passed: Forward pass returns the correct shapes")


def test_padding_mask_behaviour(cross_attention):
    '''
    Test if the padding mask is applied correctly.
    '''
    print("Testing padding mask behaviour ...")
    # Create an instance
    d_model = 10
    num_heads = 2
    dropout = 0.1
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout)

    # Create random input tensors
    batch_size = 4
    dec_seq_length = 8
    enc_seq_length = 6
    to_pad = 2
    decoder_input  = torch.randn(batch_size, dec_seq_length, d_model)
    encoder_output = torch.randn(batch_size, enc_seq_length, d_model)

    # Create a padding mask tensor with the last tokens masked
    pad_mask_enc = torch.zeros(batch_size, enc_seq_length)
    pad_mask_enc[:, to_pad:] = 1  # Mask last tokens
    pad_mask_enc = pad_mask_enc.to(torch.bool)

    # Forward pass
    _, attn_weights = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    assert torch.all(attn_weights[:, :, to_pad:] == 0), "Attention weights for padded positions should be zero"
    print("Test Passed: Padding mask is applied correctly")


def test_cross_attention_behaviour(cross_attention):
    '''
    Test if the cross-attention mechanism correctly attends to encoder outputs.
    '''
    print("Testing cross-attention behaviour ...")
    d_model = 4
    num_heads = 1
    dropout = 0.0
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout)

    # Initialize the MHA weights
    with torch.no_grad():
        # Set up Q, K, V projections 
        q_weight = torch.eye(d_model)
        k_weight = torch.eye(d_model)
        v_weight = torch.eye(d_model)
        
        # Combine them for in_proj_weight
        model.mha.in_proj_weight.data = torch.cat([q_weight, k_weight, v_weight], dim=0)
        model.mha.in_proj_bias.data.zero_()
        
        # Initialize output projection
        model.mha.out_proj.weight.data = torch.eye(d_model)
        model.mha.out_proj.bias.data.zero_()

    batch_size     = 1
    dec_seq_length = 3
    enc_seq_length = 4
    
    # Create distinct patterns for encoder output and decoder input
    encoder_output = torch.zeros(batch_size, enc_seq_length, d_model)
    encoder_output[0, 0] = torch.tensor([1.0, -1.0, -1.0, -1.0])  # Position 0
    encoder_output[0, 1] = torch.tensor([-1.0, 1.0, -1.0, -1.0])  # Position 1
    encoder_output[0, 2] = torch.tensor([-1.0, -1.0, 1.0, -1.0])  # Position 2
    encoder_output[0, 3] = torch.tensor([-1.0, -1.0, -1.0, 1.0])  # Position 3

    decoder_input = torch.zeros(batch_size, dec_seq_length, d_model)
    decoder_input[0, 0] = torch.tensor([1.0, -1.0, -1.0, -1.0])  # Should attend to position 0
    decoder_input[0, 1] = torch.tensor([-1.0, 1.0, -1.0, -1.0])  # Should attend to position 1
    decoder_input[0, 2] = torch.tensor([-1.0, -1.0, 1.0, -1.0])  # Should attend to position 2

    pad_mask_enc = torch.zeros(batch_size, enc_seq_length, dtype=torch.bool)

    # Forward pass
    _, attn_weights = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    # Test that each decoder position primarily attends to its corresponding encoder position
    for i in range(dec_seq_length):
        # Check that the maximum attention weight is at the expected position
        max_attention_pos = torch.argmax(attn_weights[0, i])
        assert max_attention_pos.item() == i, f"Position {i} attended most to position {max_attention_pos.item()} instead of position {i}"
        
        # Check that the attention is significantly higher for the correct position
        max_attention = attn_weights[0, i, i]
        other_positions_attention = attn_weights[0, i, [j for j in range(enc_seq_length) if j != i]]
        assert torch.all(max_attention > other_positions_attention), f"Position {i} does not have significantly higher attention to its corresponding position"

    print("Test Passed: Cross-attention behavior is correct")


def test_cross_attention_residual(cross_attention):
    '''
    Test if the residual connection is applied correctly.
    '''
    print("Testing residual connection ...")
    # Create an instance with no dropout for deterministic behavior
    d_model = 4
    num_heads = 2
    dropout = 0.0  # Set to 0 for deterministic behavior
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout)

    # Force MHA to be an identity transformation
    with torch.no_grad():
        model.mha.in_proj_weight.data = torch.eye(3 * d_model, d_model)
        model.mha.in_proj_bias.data.zero_()
        model.mha.out_proj.weight.data = torch.eye(d_model)
        model.mha.out_proj.bias.data.zero_()

    # Create random input tensors
    batch_size = 4
    dec_seq_length = 10
    enc_seq_length = 10
    decoder_input = torch.randn(batch_size, dec_seq_length, d_model)
    encoder_output = decoder_input.clone()  # Use same tensor to ensure identity mapping

    # Create mask (all False to allow attention)
    pad_mask_enc = torch.zeros(batch_size, enc_seq_length, dtype=torch.bool)

    # Get the output
    output, _ = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    assert torch.allclose(output, decoder_input, rtol=1e-5, atol=1e-5), "Residual connection is not applied correctly."
    print("Test Passed: Residual connection is applied correctly")


def main():
    from hw4lib.model import CrossAttentionLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'CrossAttentionLayer': [
                {   
                    'func': lambda: test_sublayer_crossattention(CrossAttentionLayer),
                    'description': 'Test the cross-attention sublayer'
                }
            ]
        }   
    )   

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()  

