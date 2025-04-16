import torch

def test_decoderlayer_selfattention(decoder_layer):
    '''
    Test the decoder layer self-attention mechanism.
    '''
    # Structural Test
    test_initialization(decoder_layer)

    # Integration Test
    test_forward_shapes(decoder_layer)
    test_sublayer_integration(decoder_layer)


def test_initialization(decoder_layer):
    '''
    Test if the sublayers exist and are properly initialized.
    '''
    print("Testing initialization ...")
    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.1
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Check if sublayers exist
    expected_attributes = {"self_attn", "ffn"}
    assert expected_attributes.issubset(dir(model)), "Required sublayers are missing"

    # Check if sublayers are not None
    assert model.self_attn is not None, "self_attn sublayer is None"
    assert model.ffn is not None, "ffn sublayer is None"

    print("Test Passed: All sublayers exist and are initialized correctly")


def test_forward_shapes(decoder_layer):
    '''
    Test the shapes of the output of the decoder layer.
    '''
    print("Testing forward shapes ...")
    
    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.0  # For deterministic testing
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Create test inputs
    batch_size = 2
    seq_length = 8
    x = torch.randn(batch_size, seq_length, d_model)
    pad_mask = torch.ones((batch_size, seq_length))
    slf_attn_mask = torch.ones((seq_length, seq_length))

    # Forward pass
    output, attn_weights = model(x, pad_mask, slf_attn_mask)

    # Check shapes
    assert output.shape == (batch_size, seq_length, d_model), \
        f"Output shape mismatch: expected {(batch_size, seq_length, d_model)} but got {output.shape}"
    assert attn_weights.shape == (batch_size, seq_length, seq_length), \
        f"Attention weights shape mismatch: expected {(batch_size, seq_length, seq_length)} but got {attn_weights.shape}"

    print("Test Passed: Forward shapes are as expected")



def test_sublayer_integration(decoder_layer):
    '''
    Test the integration of the sublayers.
    '''
    print("Testing sublayer interaction ...")
    
    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.0  # For deterministic testing
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Create test inputs
    batch_size = 2
    seq_length = 8
    x = torch.randn(batch_size, seq_length, d_model)
    pad_mask = torch.ones((batch_size, seq_length))
    slf_attn_mask = torch.ones((seq_length, seq_length))

    # Track intermediate outputs
    self_attn_output = None
    ffn_output = None
    
    def hook_self_attn(module, input_tensor, output_tensor):
        nonlocal self_attn_output
        self_attn_output = output_tensor[0]  # First element because sublayer1 returns tuple
        return output_tensor

    def hook_ffn(module, input_tensor, output_tensor):
        nonlocal ffn_output
        ffn_output = output_tensor
        return output_tensor

    # Register hooks
    handle1 = model.self_attn.register_forward_hook(hook_self_attn)
    handle3 = model.ffn.register_forward_hook(hook_ffn)

    # Forward pass
    output, _ = model(x, pad_mask, slf_attn_mask)

    # Remove hooks
    handle1.remove()
    handle3.remove()

    # Verify data flow
    assert self_attn_output is not None, "self_attn was not called"
    assert ffn_output is not None, "ffn was not called"
    assert not torch.equal(self_attn_output, x), "self_attn did not transform the input"
    assert not torch.equal(ffn_output, self_attn_output), "ffn did not transform self_attn's output"
    assert torch.equal(output, ffn_output), "Final output should match ffn's output"

    print("Test Passed: Sublayers interact correctly")   

def main():
    '''
    Main function to run the tests.
    '''
    from hw4lib.model import SelfAttentionDecoderLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'SelfAttentionDecoderLayer': [
                {
                    'func': lambda: test_decoderlayer_selfattention(SelfAttentionDecoderLayer),
                    'description': 'Test the self-attention decoder layer'
                }
            ]
        }
    )      

    framework.run_tests()
    framework.summarize_results()

if __name__ == "__main__":
    main()  