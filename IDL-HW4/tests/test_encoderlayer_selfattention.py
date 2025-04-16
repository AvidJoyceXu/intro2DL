import torch

def test_encoderlayer_selfattention(encoder_layer):
    '''
    Test the encoder layer self-attention mechanism.
    '''
    # Structural Test
    test_initialization(encoder_layer)

    # Integration Test
    test_forward_shapes(encoder_layer)
    test_sublayer_integration(encoder_layer)
    test_bidirectional_attention(encoder_layer)


def test_initialization(encoder_layer):
    '''
    Test if the sublayers exist and are properly initialized.
    '''
    print("Testing initialization ...")
    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.1
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Check if sublayers exist
    expected_attributes = {"self_attn", "ffn"}
    assert expected_attributes.issubset(dir(model)), "Required sublayers are missing"

    # Check if sublayers are not None
    assert model.self_attn is not None, "self_attn sublayer is None"
    assert model.ffn is not None, "ffn sublayer is None"

    print("Test Passed: All sublayers exist and are initialized correctly")


def test_forward_shapes(encoder_layer):
    '''
    Test the shapes of the output of the encoder layer.
    '''
    print("Testing forward shapes ...")
    
    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.0  # For deterministic testing
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Create test inputs
    batch_size = 2
    seq_length = 8
    x = torch.randn(batch_size, seq_length, d_model)
    pad_mask = torch.ones((batch_size, seq_length))

    # Forward pass
    output, attn_weights = model(x, pad_mask)

    # Check shapes
    assert output.shape == (batch_size, seq_length, d_model), \
        f"Output shape mismatch: expected {(batch_size, seq_length, d_model)} but got {output.shape}"
    assert attn_weights.shape == (batch_size, seq_length, seq_length), \
        f"Attention weights shape mismatch: expected {(batch_size, seq_length, seq_length)} but got {attn_weights.shape}"

    print("Test Passed: Forward shapes are as expected")


def test_sublayer_integration(encoder_layer):
    '''
    Test the integration of the sublayers.
    '''
    print("Testing sublayer interaction ...")
    
    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.0  # For deterministic testing
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Create test inputs
    batch_size = 2
    seq_length = 8
    x = torch.randn(batch_size, seq_length, d_model)
    pad_mask = torch.ones((batch_size, seq_length))

    # Track intermediate outputs
    self_attn_output = None
    ffn_output = None
    
    def hook_self_attn(module, input_tensor, output_tensor):
        nonlocal self_attn_output
        self_attn_output = output_tensor[0]  # First element because sublayer returns tuple
        return output_tensor

    def hook_ffn(module, input_tensor, output_tensor):
        nonlocal ffn_output
        ffn_output = output_tensor
        return output_tensor

    # Register hooks
    handle1 = model.self_attn.register_forward_hook(hook_self_attn)
    handle2 = model.ffn.register_forward_hook(hook_ffn)

    # Forward pass
    output, _ = model(x, pad_mask)

    # Remove hooks
    handle1.remove()
    handle2.remove()

    # Verify data flow
    assert self_attn_output is not None, "self_attn was not called"
    assert ffn_output is not None, "ffn was not called"
    assert not torch.equal(self_attn_output, x), "self_attn did not transform the input"
    assert not torch.equal(ffn_output, self_attn_output), "ffn did not transform self_attn's output"
    assert torch.equal(output, ffn_output), "Final output should match ffn's output"

    print("Test Passed: Sublayers interact correctly")


def test_bidirectional_attention(encoder_layer):
    '''
    Test that the encoder can attend to all positions (bidirectional attention).
    '''
    print("Testing bidirectional attention ...")
    
    d_model = 4
    num_heads = 1
    d_ff = 8
    dropout = 0.0
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Initialize the MHA weights for deterministic attention
    with torch.no_grad():
        model.self_attn.mha.in_proj_weight.data = torch.eye(3 * d_model, d_model)
        model.self_attn.mha.in_proj_bias.data.zero_()
        model.self_attn.mha.out_proj.weight.data = torch.eye(d_model)
        model.self_attn.mha.out_proj.bias.data.zero_()

    # Create input with distinct patterns
    batch_size = 1
    seq_length = 4
    x = torch.zeros(batch_size, seq_length, d_model)
    x[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Position 0
    x[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # Position 1
    x[0, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])  # Position 2
    x[0, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # Position 3

    # No padding mask
    pad_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

    # Forward pass
    _, attn_weights = model(x, pad_mask)

    # Test that each position can attend to all other positions
    for i in range(seq_length):
        for j in range(seq_length):
            assert attn_weights[0, i, j] > 0, \
                f"Position {i} cannot attend to position {j}"

    # Test that attention weights for each query sum to 1
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1))), \
        "Attention weights do not sum to 1"

    print("Test Passed: Bidirectional attention is working correctly")


def main():
    '''
    Main function to run the tests.
    '''
    from hw4lib.model import SelfAttentionEncoderLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'SelfAttentionEncoderLayer': [
                {
                    'func': lambda: test_encoderlayer_selfattention(SelfAttentionEncoderLayer),
                    'description': 'Test the self-attention encoder layer'
                }
            ]
        }
    )      

    framework.run_tests()
    framework.summarize_results()

if __name__ == "__main__":
    main()