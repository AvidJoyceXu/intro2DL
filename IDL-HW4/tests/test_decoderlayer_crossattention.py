import torch

def test_decoderlayer_crossattention(decoder_layer):
    '''
    Test the decoder layer with both self-attention and cross-attention mechanisms integrated.   
    '''
    # Structural Test
    test_initialization(decoder_layer)

    # Integration Test
    test_forward_shapes(decoder_layer)  
    test_sublayer_integration(decoder_layer)
    test_cross_attention_integration(decoder_layer)

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

    # Check if sublayers exist (including sublayer2 which is unique to DecoderLayer2)
    expected_attributes = {"self_attn", "cross_attn", "ffn"}
    assert expected_attributes.issubset(dir(model)), "Required sublayers are missing"

    # Check if sublayers are not None
    assert model.self_attn is not None, "self_attn sublayer is None"
    assert model.cross_attn is not None, "cross_attn sublayer is None"
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
    dec_seq_length = 8
    enc_seq_length = 10
    x = torch.randn(batch_size, dec_seq_length, d_model)
    enc_output = torch.randn(batch_size, enc_seq_length, d_model)
    pad_mask_dec = torch.ones(batch_size, dec_seq_length)
    pad_mask_enc = torch.ones(batch_size, enc_seq_length)
    slf_attn_mask = torch.ones(dec_seq_length, dec_seq_length)

    # Forward pass
    output, self_attn_weights, cross_attn_weights = model(
        x, enc_output, pad_mask_dec, pad_mask_enc, slf_attn_mask
    )

    # Check shapes
    assert output.shape == (batch_size, dec_seq_length, d_model), \
        f"Output shape mismatch: expected {(batch_size, dec_seq_length, d_model)} but got {output.shape}"
    assert self_attn_weights.shape == (batch_size, dec_seq_length, dec_seq_length), \
        f"Self attention weights shape mismatch: expected {(batch_size, dec_seq_length, dec_seq_length)} but got {self_attn_weights.shape}"
    assert cross_attn_weights.shape == (batch_size, dec_seq_length, enc_seq_length), \
        f"Cross attention weights shape mismatch: expected {(batch_size, dec_seq_length, enc_seq_length)} but got {cross_attn_weights.shape}"

    print("Test Passed: Forward shapes are as expected")


def test_sublayer_integration(decoder_layer):
    '''
    Test the integration of the sublayers.
    '''
    print("Testing sublayer integration ...")

    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.0  # For deterministic testing
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Create test inputs
    batch_size = 2
    dec_seq_length = 8
    enc_seq_length = 10
    x = torch.randn(batch_size, dec_seq_length, d_model)
    enc_output = torch.randn(batch_size, enc_seq_length, d_model)
    pad_mask_dec  = torch.zeros(batch_size, dec_seq_length)
    pad_mask_enc  = torch.zeros(batch_size, enc_seq_length)
    slf_attn_mask = torch.zeros(dec_seq_length, dec_seq_length)

    # Track intermediate outputs
    sublayer_outputs = {}
    def make_hook(name):
        def hook(module, input_tensor, output_tensor):
            sublayer_outputs[name] = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            return output_tensor
        return hook

    # Register hooks
    handles = [
        model.self_attn.register_forward_hook(make_hook('self_attn')),
        model.cross_attn.register_forward_hook(make_hook('cross_attn')),
        model.ffn.register_forward_hook(make_hook('ffn'))
    ]

    # Forward pass
    output, _, _ = model(x, enc_output, pad_mask_dec, pad_mask_enc, slf_attn_mask)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Verify data flow
    assert len(sublayer_outputs) == 3, "Not all sublayers were called"
    assert not torch.equal(sublayer_outputs['self_attn'], x), "self_attn did not transform the input"
    assert not torch.equal(sublayer_outputs['cross_attn'], sublayer_outputs['self_attn']), \
        "cross_attn did not transform self_attn's output"
    assert not torch.equal(sublayer_outputs['cross_attn'], sublayer_outputs['ffn']), \
        "ffn did not transform cross_attn's output"
    assert torch.equal(output, sublayer_outputs['ffn']), \
        "ffn is not the final output"
    
    
    print("Test Passed: Sublayers interact correctly")



def test_cross_attention_integration(decoder_layer):
    '''
    Test the integration of the cross-attention mechanism.
    '''
    print("Testing cross-attention behavior ...")
    
    d_model = 16
    num_heads = 4
    d_ff = 32
    dropout = 0.0
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

    # Create test inputs
    batch_size = 2
    dec_seq_length = 8
    enc_seq_length = 10
    x = torch.randn(batch_size, dec_seq_length, d_model)
    pad_mask_dec = torch.ones(batch_size, dec_seq_length)
    pad_mask_enc = torch.ones(batch_size, enc_seq_length)
    slf_attn_mask = torch.ones(dec_seq_length, dec_seq_length)

    # Test 1: Different encoder outputs should produce different results
    enc_output1 = torch.randn(batch_size, enc_seq_length, d_model)
    enc_output2 = torch.randn(batch_size, enc_seq_length, d_model)

    output1, _, cross_attn1 = model(x, enc_output1, pad_mask_dec, pad_mask_enc, slf_attn_mask)
    output2, _, cross_attn2 = model(x, enc_output2, pad_mask_dec, pad_mask_enc, slf_attn_mask)

    assert not torch.allclose(output1, output2), "Different encoder outputs should produce different results"
    assert not torch.allclose(cross_attn1, cross_attn2), "Different encoder outputs should produce different attention patterns"

    # Test 2: Masked encoder positions should be ignored
    pad_mask_enc_masked = torch.zeros(batch_size, enc_seq_length, dtype=torch.bool)
    pad_mask_enc_masked[:, -5:] = True  # Mask last 5 positions

    _, _, cross_attn_masked = model(x, enc_output1, pad_mask_dec, pad_mask_enc_masked, slf_attn_mask)
    
    assert torch.all(cross_attn_masked[:, :, -5:] == 0), "Masked encoder positions should be ignored in cross-attention"

    print("Test Passed: Cross-attention behaves correctly")


def main():
    '''
    Main function to run the tests.
    '''
    from hw4lib.model import CrossAttentionDecoderLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'CrossAttentionDecoderLayer': [
                {
                    'func': lambda: test_decoderlayer_crossattention(CrossAttentionDecoderLayer),
                    'description': 'Test the cross-attention decoder layer'
                }
            ]
        }
    )      
    framework.run_tests()
    framework.summarize_results()

if __name__ == "__main__":
    main()