import torch 


def test_decoder_only_transformer(transformer):
    '''
    Test the decoder only transformer
    '''
    
    # Structural tests
    test_initialization(transformer)

    # Integration tests
    test_forward_pass(transformer)
    test_forward_propagation_order(transformer)


def test_initialization(transformer):
    '''
    Test the initialization of the decoder only transformer
    '''
    print("Testing initialization...")
    
    # Test parameters
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2
    dropout = 0.1
    max_len = 100
    num_classes = 10
    
    model = transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes
    )

    # Check if all required components exist
    expected_attributes = {
        "dec_layers", "target_embedding", "positional_encoding",
        "final_linear", "dropout", "norm", "max_len", "num_classes", "num_layers"
    }
    assert expected_attributes.issubset(dir(model)), "Required components are missing"

    # Check all attributes are not None
    for attr in expected_attributes:
        assert getattr(model, attr) is not None, f"{attr} is None"  
    
    # Check number of decoder layers
    assert len(model.dec_layers) == num_layers, f"Expected {num_layers} decoder layers, got {len(model.dec_layers)}"
    
    # Check embedding dimensions
    assert model.target_embedding.embedding_dim == d_model, "Embedding dimension mismatch"
    assert model.target_embedding.num_embeddings == num_classes, "Vocabulary size mismatch"
    
    print("Test Passed: All components initialized correctly")



def test_forward_pass(transformer):
    '''
    Test the forward pass of the decoder only transformer
    '''
    print("Testing forward pass...")
    
    # Test parameters
    batch_size = 2
    seq_length = 8
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2
    dropout = 0.0  # For deterministic testing
    max_len = 100
    num_classes = 10
    
    model = transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes
    )
    
    # Create test inputs
    targets = torch.randint(0, num_classes, (batch_size, seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * seq_length
    
    # Forward pass
    output, attention_weights = model(targets, target_lengths)
    
    # Check output shape
    expected_shape = (batch_size, seq_length, num_classes)
    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    
    # Check attention weights
    for i in range(num_layers):
        attn_key = f'layer{i+1}_dec_self'
        assert attn_key in attention_weights, f"Missing attention weights for {attn_key}"
        assert attention_weights[attn_key].shape == (batch_size, seq_length, seq_length), \
            f"Attention weights shape mismatch for {attn_key}"
    
    print("Test Passed: Forward pass works correctly")


def test_forward_propagation_order(transformer):
    '''
    Test that modules are called in the correct order during forward propagation.
    '''
    print("Testing forward propagation order...")
    
    # Test parameters
    batch_size = 2
    seq_length = 8
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2
    
    model = transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=100,
        num_classes=10
    )
    
    # Create test inputs
    targets = torch.randint(0, 10, (batch_size, seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * seq_length
    
    # Track module execution order
    execution_order = []
    
    def make_hook(name):
        def hook(module, input_tensor, output_tensor):
            execution_order.append(name)
            return output_tensor
        return hook
    
    # Register hooks for all relevant modules
    handles = [
        model.target_embedding.register_forward_hook(make_hook('target_embedding')),
        model.positional_encoding.register_forward_hook(make_hook('pos_encoding')),
        model.dropout.register_forward_hook(make_hook('dropout')),
    ]
    
    # Add hooks for decoder layers
    for i, layer in enumerate(model.dec_layers):
        handles.append(layer.register_forward_hook(make_hook(f'decoder_layer_{i}')))
    
    # Add hooks for final operations
    handles.extend([
        model.norm.register_forward_hook(make_hook('final_norm')),
        model.final_linear.register_forward_hook(make_hook('final_linear'))
    ])
    
    # Forward pass
    model(targets, target_lengths)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Expected order based on implementation
    expected_order = [
        'target_embedding',
        'pos_encoding',
        'dropout',
        'decoder_layer_0',
        'decoder_layer_1',
        'final_norm',
        'final_linear'
    ]
    
    # Verify execution order
    assert execution_order == expected_order, \
        f"Incorrect execution order.\nExpected: {expected_order}\nGot: {execution_order}"
    
    print("Test Passed: Forward propagation order is correct")    


def main():
    '''
    Main function to run the tests
    '''
    from hw4lib.model import DecoderOnlyTransformer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'DecoderOnlyTransformer': [
                {
                    'func': lambda: test_decoder_only_transformer(DecoderOnlyTransformer),
                    'description': 'Test the decoder only transformer'
                }
            ]
        }
    )
    framework.run_tests()
    framework.summarize_results()   

if __name__ == '__main__':
    main()  