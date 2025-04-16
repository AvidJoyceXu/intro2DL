import torch
from hw4lib.model import PadMask
def test_encoder_decoder_transformer(transformer):
    '''
    Test the EncoderDecoderTransformer implementation.
    '''
    # Structural Tests
    test_initialization(transformer)
    
    # Integration Tests
    test_encode_method(transformer)
    test_decode_method(transformer)
    test_forward_pass(transformer)
    test_encoder_decoder_integration(transformer)
    test_ctc_integration(transformer)
    test_forward_propagation_order(transformer)

def test_initialization(transformer):
    '''
    Test if the transformer is properly initialized.
    '''
    print("Testing initialization...")
    
    # Test parameters
    input_dim = 80
    time_reduction = 2
    reduction_method = 'both'
    d_model = 16
    num_encoder_heads = 4
    num_decoder_heads = 4
    d_ff_encoder = 32
    d_ff_decoder = 32
    dropout = 0.1
    max_len = 100
    num_classes = 10
    num_encoder_layers = 2
    num_decoder_layers = 2
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=time_reduction,
        reduction_method=reduction_method,
        num_encoder_layers=num_encoder_layers,
        num_encoder_heads=num_encoder_heads,
        d_ff_encoder=d_ff_encoder,
        num_decoder_layers=num_decoder_layers,
        num_decoder_heads=num_decoder_heads,
        d_ff_decoder=d_ff_decoder,
        d_model=d_model,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes
    )

    # Check if all required components exist
    expected_attributes = {
        "enc_layers", "dec_layers", "source_embedding", "target_embedding",
        "positional_encoding", "final_linear", "encoder_norm", "decoder_norm",
        "ctc_head"
    }
    assert expected_attributes.issubset(dir(model)), "Required components are missing"
    
    # Check number of layers
    assert len(model.enc_layers) == num_encoder_layers, f"Expected {num_encoder_layers} encoder layers, got {len(model.enc_layers)}"
    assert len(model.dec_layers) == num_decoder_layers, f"Expected {num_decoder_layers} decoder layers, got {len(model.dec_layers)}"
    
    # Check embedding dimensions
    assert model.target_embedding.embedding_dim == d_model, "Target embedding dimension mismatch"
    assert model.target_embedding.num_embeddings == num_classes, "Target vocabulary size mismatch"
    
    print("Test Passed: All components initialized correctly")

def test_forward_pass(transformer):
    '''
    Test the forward pass of the transformer.
    '''
    print("Testing forward pass...")
    
    # Test parameters
    batch_size = 2
    input_seq_length = 100
    target_seq_length = 8
    input_dim = 80
    time_reduction = 2
    reduction_method = 'both'
    d_model = 16
    num_encoder_heads = 4
    num_decoder_heads = 4
    d_ff_encoder = 32
    d_ff_decoder = 32
    dropout = 0.0  # For deterministic testing
    max_len = 100
    num_classes = 10
    num_encoder_layers = 2
    num_decoder_layers = 2

    model = transformer(
        input_dim=input_dim,
        time_reduction=time_reduction,
        reduction_method=reduction_method,
        num_encoder_layers=num_encoder_layers,
        num_encoder_heads=num_encoder_heads,
        d_ff_encoder=d_ff_encoder,
        num_decoder_layers=num_decoder_layers,
        num_decoder_heads=num_decoder_heads,
        d_ff_decoder=d_ff_decoder,
        d_model=d_model,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes
    )
    
    # Create test inputs
    source = torch.randn(batch_size, input_seq_length, input_dim)
    source_lengths = torch.randint(input_seq_length // 2, input_seq_length, (batch_size,))
    targets = torch.randint(0, num_classes, (batch_size, target_seq_length))
    target_lengths = torch.randint(target_seq_length // 2, target_seq_length, (batch_size,))
    
    # Forward pass
    output, attention_weights, _ = model(source, targets, source_lengths, target_lengths)
    
    # Check output shapes
    assert output.shape == (batch_size, target_seq_length, num_classes), \
        f"Output shape mismatch: expected {(batch_size, target_seq_length, num_classes)}, got {output.shape}"
    
    # Check attention weights
    for i in range(num_encoder_layers):
        enc_attn_key = f'layer{i+1}_enc_self'
        assert enc_attn_key in attention_weights, f"Missing encoder attention weights for {enc_attn_key}"
    
    for i in range(num_decoder_layers):
        dec_self_attn_key = f'layer{i+1}_dec_self'
        dec_cross_attn_key = f'layer{i+1}_dec_cross'
        assert dec_self_attn_key in attention_weights, f"Missing decoder self-attention weights for {dec_self_attn_key}"
        assert dec_cross_attn_key in attention_weights, f"Missing decoder cross-attention weights for {dec_cross_attn_key}"
    
    print("Test Passed: Forward pass works correctly")

def test_encoder_decoder_integration(transformer):
    '''
    Test the integration between encoder and decoder components.
    '''
    print("Testing encoder-decoder integration...")
    
    # Test parameters
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    d_model = 16
    num_encoder_heads = 4
    num_decoder_heads = 4
    d_ff_encoder = 32
    d_ff_decoder = 32
    dropout = 0.0
    max_len = 100
    num_classes = 10
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=num_encoder_heads,
        d_ff_encoder=d_ff_encoder,
        num_decoder_layers=2,
        num_decoder_heads=num_decoder_heads,
        d_ff_decoder=d_ff_decoder,
        d_model=d_model,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes
    )
    
    # Create two different input sequences
    source1 = torch.randn(batch_size, input_seq_length, input_dim)
    source2 = torch.randn(batch_size, input_seq_length, input_dim)
    source_lengths = torch.ones(batch_size, dtype=torch.int32) * input_seq_length
    
    # Same target sequence
    targets = torch.randint(0, num_classes, (batch_size, target_seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * target_seq_length
    
    # Get outputs for different inputs
    output1, attn1, ctc1 = model(source1, targets, source_lengths, target_lengths)
    output2, attn2, ctc2 = model(source2, targets, source_lengths, target_lengths)
    
    # Outputs should be different for different inputs
    assert not torch.allclose(output1, output2), "Different inputs should produce different outputs"
    assert not torch.allclose(ctc1['log_probs'], ctc2['log_probs']), "Different inputs should produce different CTC outputs"
    
    # Cross-attention patterns should be different
    for i in range(model.num_decoder_layers):
        cross_attn_key = f'layer{i+1}_dec_cross'
        assert not torch.allclose(attn1[cross_attn_key], attn2[cross_attn_key]), \
            f"Cross-attention patterns should differ for layer {i+1}"
    
    print("Test Passed: Encoder-decoder integration works correctly")

def test_ctc_integration(transformer):
    '''
    Test the CTC head integration.
    '''
    print("Testing CTC integration...")
    
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=16,
        dropout=0.0,
        max_len=100,
        num_classes=10
    )
    
    # Create inputs
    source = torch.randn(batch_size, input_seq_length, input_dim)
    source_lengths = torch.ones(batch_size, dtype=torch.int32) * input_seq_length
    targets = torch.randint(0, 10, (batch_size, target_seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * target_seq_length
    
    # Forward pass
    _, _, ctc_input = model(source, targets, source_lengths, target_lengths)
    
    # Check CTC output properties
    assert ctc_input['log_probs'].dim() == 3, "CTC output should be 3-dimensional"
    assert ctc_input['log_probs'].shape[2] == model.num_classes, "CTC output should have num_classes as last dimension"
    
    # Test that LogSoftmax has been applied to the output
    ctc_probs = torch.exp(ctc_input['log_probs'])
    assert torch.allclose(ctc_probs.sum(dim=-1), torch.ones_like(ctc_probs.sum(dim=-1))), \
        "CTC probabilities should sum to 1"
    
    print("Test Passed: CTC integration works correctly")

def test_forward_propagation_order(transformer):
    '''
    Test that modules are called in the correct order during forward propagation.
    '''
    print("Testing forward propagation order...")
    
    # Test parameters
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=16,
        dropout=0.0,
        max_len=100,
        num_classes=10
    )
    
    # Create test inputs
    source = torch.randn(batch_size, input_seq_length, input_dim)
    source_lengths = torch.ones(batch_size, dtype=torch.int32) * input_seq_length
    targets = torch.randint(0, 10, (batch_size, target_seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * target_seq_length
    
    # Track module execution order
    execution_order = []
    
    def make_hook(name):
        def hook(module, input_tensor, output_tensor):
            execution_order.append(name)
            return output_tensor
        return hook
    
    # Register hooks for all relevant modules
    handles = [
        model.source_embedding.register_forward_hook(make_hook('source_embedding')),
        model.target_embedding.register_forward_hook(make_hook('target_embedding')),
        model.positional_encoding.register_forward_hook(make_hook('pos_encoding')),
        model.dropout.register_forward_hook(make_hook('dropout')),
    ]
    
    # Add hooks for encoder layers
    for i, layer in enumerate(model.enc_layers):
        handles.append(layer.register_forward_hook(make_hook(f'encoder_layer_{i}')))
    
    # Add hooks for decoder layers
    for i, layer in enumerate(model.dec_layers):
        handles.append(layer.register_forward_hook(make_hook(f'decoder_layer_{i}')))
    
    # Add hooks for final operations
    handles.extend([
        model.encoder_norm.register_forward_hook(make_hook('encoder_norm')),
        model.decoder_norm.register_forward_hook(make_hook('decoder_norm')),
        model.final_linear.register_forward_hook(make_hook('final_linear')),
        model.ctc_head.register_forward_hook(make_hook('ctc_head'))
    ])
    
    # Forward pass
    model(source, targets, source_lengths, target_lengths)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Expected order based on implementation
    expected_order = [
        'source_embedding',
        "pos_encoding",
        "dropout",
        'encoder_layer_0',
        'encoder_layer_1',
        'encoder_norm',
        "ctc_head",
        'target_embedding',
        'pos_encoding',
        'dropout',
        'decoder_layer_0',
        'decoder_layer_1',
        'decoder_norm',
        'final_linear',
    ]
    
    # Check execution order
    assert execution_order == expected_order, \
        f"Incorrect execution order. Expected {expected_order}, got {execution_order}"
    
    print("Test Passed: Forward propagation order is correct")

def test_encode_method(transformer):
    '''
    Test the encode method of the transformer.
    '''
    print("Testing encode method...")
    
    # Test parameters
    batch_size = 2
    input_seq_length = 50
    input_dim = 80
    d_model = 16
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=d_model,
        dropout=0.0,
        max_len=100,
        num_classes=10
    )
    
    # Create test inputs
    source = torch.randn(batch_size, input_seq_length, input_dim)
    source_lengths =  torch.randint(input_seq_length // 2, input_seq_length, (batch_size,))
    
    # Test encode method
    encoder_output, pad_mask_src, encoder_attention, ctc_input = model.encode(source, source_lengths)
    
    # Calculate expected encoder output length
    expected_enc_length = model.source_embedding.calculate_downsampled_length(torch.ones(batch_size, dtype=torch.int32) * input_seq_length)
    expected_enc_length = expected_enc_length[0].item()
    
    # Check shapes
    assert encoder_output.shape == (batch_size, expected_enc_length, d_model), \
        f"Encoder output shape mismatch: expected {(batch_size, expected_enc_length, d_model)}, got {encoder_output.shape}"
    assert torch.equal(
        pad_mask_src, 
        PadMask(encoder_output, model.source_embedding.calculate_downsampled_length(source_lengths))
    ), "Pad mask source mismatch"

    assert ctc_input['log_probs'].shape == (expected_enc_length, batch_size, model.num_classes), \
        f"CTC output shape mismatch: expected {(expected_enc_length, batch_size, model.num_classes)}, got {ctc_input['log_probs'].shape}"
    
    # Check encoder attention weights
    for i in range(model.num_encoder_layers):
        attn_key = f'layer{i+1}_enc_self'
        assert attn_key in encoder_attention, f"Missing encoder attention weights for {attn_key}"
        assert encoder_attention[attn_key].shape == (batch_size, expected_enc_length, expected_enc_length), \
            f"Encoder attention weights shape mismatch for {attn_key}"
    
    # Test that LogSoftmax has been applied to the output
    ctc_probs = torch.exp(ctc_input['log_probs'])
    assert torch.allclose(ctc_probs.sum(dim=-1), torch.ones_like(ctc_probs.sum(dim=-1))), \
        "CTC probabilities should sum to 1"
    
    print("Test Passed: Encode method works correctly")

def test_decode_method(transformer):
    '''
    Test the decode method of the transformer.
    '''
    print("Testing decode method...")
    
    # Test parameters
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    d_model = 16
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=d_model,
        dropout=0.0,
        max_len=100,
        num_classes=10
    )
    
    # First get encoder output
    source = torch.randn(batch_size, input_seq_length, input_dim)
    source_lengths = torch.randint(input_seq_length // 2, input_seq_length, (batch_size,))
    encoder_output, pad_mask_src, _, _ = model.encode(source, source_lengths)
    
    # Create decoder inputs
    targets = torch.randint(0, 10, (batch_size, target_seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * target_seq_length
    
    # Test decode method
    decoder_output, decoder_attention = model.decode(targets, encoder_output, target_lengths, pad_mask_src)
    
    # Check shapes
    assert decoder_output.shape == (batch_size, target_seq_length, model.num_classes), \
        f"Decoder output shape mismatch: expected {(batch_size, target_seq_length, model.num_classes)}, got {decoder_output.shape}"
    
    # Check decoder attention weights
    expected_enc_length = model.source_embedding.calculate_downsampled_length(torch.ones(batch_size, dtype=torch.int32) * input_seq_length)
    expected_enc_length = expected_enc_length[0].item()
    for i in range(model.num_decoder_layers):
        # Check self-attention
        self_attn_key = f'layer{i+1}_dec_self'
        assert self_attn_key in decoder_attention, f"Missing decoder self-attention weights for {self_attn_key}"
        assert decoder_attention[self_attn_key].shape == (batch_size, target_seq_length, target_seq_length), \
            f"Decoder self-attention weights shape mismatch for {self_attn_key}"
        
        # Check cross-attention
        cross_attn_key = f'layer{i+1}_dec_cross'
        assert cross_attn_key in decoder_attention, f"Missing decoder cross-attention weights for {cross_attn_key}"
        assert decoder_attention[cross_attn_key].shape == (batch_size, target_seq_length, expected_enc_length), \
            f"Decoder cross-attention weights shape mismatch for {cross_attn_key}"
    
    # Test causal masking in self-attention
    for i in range(model.num_decoder_layers):
        self_attn_key = f'layer{i+1}_dec_self'
        attn_weights = decoder_attention[self_attn_key]
        
        # Check that future positions are masked (upper triangular should be 0)
        for t in range(target_seq_length):
            for future_t in range(t + 1, target_seq_length):
                assert torch.all(attn_weights[:, t, future_t] == 0), \
                    f"Position {t} should not attend to future position {future_t}"
    
    print("Test Passed: Decode method works correctly")

def main():
    '''
    Main function to run the tests
    '''
    from hw4lib.model import EncoderDecoderTransformer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'EncoderDecoderTransformer': [
                {
                    'func': lambda: test_encoder_decoder_transformer(EncoderDecoderTransformer),
                    'description': 'Test the encoder-decoder transformer'
                }
            ]
        }
    )
    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
