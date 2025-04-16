import torch
import torch.nn as nn

def test_sublayer_feedforward(feedforward):
    '''
    Test the feedforward sublayer.
    Args:
        feedforward (nn.Module): The feedforward sublayer.  
    '''
    # Structural Test
    test_initialization(feedforward)

    # Functional Tests
    test_forward_shapes(feedforward)
    
    # Behavioral Tests
    test_ffn_behavior(feedforward)
    test_residual_connection(feedforward)
    test_layer_norm(feedforward)
    test_forward_order(feedforward)

def test_initialization(feedforward):
    '''
    Test if the layers exist and match the reference implementation.
    '''
    print("Testing initialization ...")
    d_model = 10
    d_ff    = 40
    dropout = 0.1
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    # Check if the layers exist in the model
    expected_attributes = {"ffn", "norm", "dropout"}
    assert expected_attributes.issubset(dir(model)), "Required attributes are missing"

    # Check if the layers are not None
    assert model.ffn is not None, "Feed-forward network is None"
    assert model.norm is not None, "Normalization layer is None"
    assert model.dropout is not None, "Dropout layer is None"   

    # Check if the layers are of the correct type
    assert isinstance(model.ffn, nn.Sequential), "Feed-forward network is not of the correct type"
    assert isinstance(model.norm, nn.LayerNorm), "Normalization layer is not of the correct type"
    assert isinstance(model.dropout, nn.Dropout), "Dropout layer is not of the correct type"

    # Check FFN structure
    assert len(model.ffn) == 4, "FFN should have exactly 4 layers"
    assert isinstance(model.ffn[0], nn.Linear), "First FFN layer should be Linear"
    assert isinstance(model.ffn[1], (nn.GELU, nn.ReLU, nn.LeakyReLU, nn.SiLU)), "Second FFN layer should be GELU or ReLU or LeakyReLU or SiLU"
    assert isinstance(model.ffn[2], nn.Dropout), "Third FFN layer should be Dropout"
    assert isinstance(model.ffn[3], nn.Linear), "Fourth FFN layer should be Linear"

    # Check dimensions
    assert model.ffn[0].in_features == d_model, f"FFN input dimension: expected {d_model} but got {model.ffn[0].in_features}"
    assert model.ffn[0].out_features == d_ff, f"FFN hidden dimension: expected {d_ff} but got {model.ffn[0].out_features}"
    assert model.ffn[3].in_features == d_ff, f"FFN input dimension: expected {d_ff} but got {model.ffn[3].in_features}"
    assert model.ffn[3].out_features == d_model, f"FFN output dimension: expected {d_model} but got {model.ffn[3].out_features}"
    
    print("Test Passed: All layers exist and match reference implementation")


def test_forward_shapes(feedforward):
    '''
    Test the forward shapes of the feedforward sublayer.
    '''
    print("Testing forward shapes ...")
    
    # Create an instance
    d_model = 10
    d_ff    = 40
    dropout = 0.1
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    # Create random input tensors of different shapes
    batch_sizes = [1, 4, 8]
    seq_lengths = [10, 20, 15]
    
    for batch_size, seq_length in zip(batch_sizes, seq_lengths):
        input_tensor = torch.randn(batch_size, seq_length, d_model)
        output = model.forward(input_tensor)
        
        assert output.shape == input_tensor.shape, \
            f"Output shape mismatch: expected {input_tensor.shape} but got {output.shape}"
    
    print("Test Passed: Forward pass returns correct shapes for various input dimensions")


def test_ffn_behavior(feedforward):
    '''
    Test the behavior of the feedforward sublayer.
    '''
    print("Testing feed-forward network behavior ...")
    
    d_model = 4
    d_ff    = 8
    dropout = 0.0  # Set to 0 for deterministic testing
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    # Create a simple input tensor
    input_tensor = torch.ones(2, 3, d_model)
    
    # Forward pass
    output = model.forward(input_tensor)
    
    # Check that output is different from input (transformed)
    assert not torch.allclose(output, input_tensor), \
        "FFN output is identical to input, suggesting no transformation"
    
    # Check that output values are within reasonable range
    assert torch.all(torch.isfinite(output)), "Output contains NaN or infinite values"
    
    print("Test Passed: Feed-forward network transforms input appropriately")


def test_residual_connection(feedforward):
    '''
    Test the residual connection of the feedforward sublayer.
    '''
    print("Testing residual connection ...")
    
    d_model = 4
    d_ff    = 8
    dropout = 0.0  # Set to 0 for deterministic testing
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    # Force FFN to be an identity transformation
    # By setting the weights and biases appropriately
    with torch.no_grad():
        # First linear layer: d_model -> d_ff
        model.ffn[0].weight.data = torch.zeros(d_ff, d_model)
        model.ffn[0].bias.data = torch.zeros(d_ff)
        
        # Second linear layer: d_ff -> d_model
        model.ffn[3].weight.data = torch.zeros(d_model, d_ff)
        model.ffn[3].bias.data = torch.zeros(d_model)
    
    # Create input tensor with specific values
    batch_size = 2
    seq_length = 3
    input_tensor = torch.randn(batch_size, seq_length, d_model)
    
    # Test 1: With identity FFN, output should equal input
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor, rtol=1e-5, atol=1e-5), \
        "Residual connection is not applied correctly with identity FFN"
    
    # Test 2: With non-zero FFN, verify both paths contribute
    # Reset FFN to non-zero values
    with torch.no_grad():
        model.ffn[0].weight.data = torch.randn(d_ff, d_model) * 0.1
        model.ffn[3].weight.data = torch.randn(d_model, d_ff) * 0.1
    
    output_with_ffn = model(input_tensor)
    
    # The output should not be identical to either the input or the FFN output alone
    ffn_only = model.ffn(model.norm(input_tensor))
    assert not torch.allclose(output_with_ffn, input_tensor, rtol=1e-3, atol=1e-3), \
        "Output is identical to input, suggesting FFN path is not active"
    assert not torch.allclose(output_with_ffn, ffn_only, rtol=1e-3, atol=1e-3), \
        "Output is identical to FFN output, suggesting residual path is not active"
    
    # Test 3: Verify the scale of the output is reasonable
    # The output should be a combination of the input and the FFN output
    assert torch.allclose(output_with_ffn, input_tensor + model.dropout(ffn_only), rtol=1e-5, atol=1e-5), \
        "Output is not the sum of input and FFN output as expected"
    
    print("Test Passed: Residual connection is working correctly")  


def test_layer_norm(feedforward):
    '''
    Test the layer normalization of the feedforward sublayer.
    '''
    print("Testing layer normalization ...")
    
    d_model = 4
    d_ff    = 8
    dropout = 0.0  # Set to 0 for deterministic testing
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    # Create input tensor
    input_tensor = torch.randn(2, 3, d_model)
    
    # Monitor the layer norm calls
    norm_called = False
    
    def forward_norm_hook(module, input_tensor, output_tensor):
        nonlocal norm_called
        norm_called = True
        return output_tensor
    
    # Register the forward hook
    hook_handle = model.norm.register_forward_hook(forward_norm_hook)
    
    # Forward pass
    _ = model.forward(input_tensor)
    
    # Remove the hook
    hook_handle.remove()
    
    # Check that normalization was called
    assert norm_called, "Layer normalization was not applied during forward pass"
    
    print("Test Passed: Layer normalization is being applied correctly")

def test_forward_order(feedforward):
    '''
    Test if the forward pass follows the correct order of operations:
    1. Create residual connection
    2. Apply normalization
    3. Apply FFN with dropout
    4. Add residual connection
    '''
    print("Testing forward pass order ...")
    
    d_model = 4
    d_ff    = 8
    dropout = 0.0  # Set to 0 for deterministic testing
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    # Create input tensor
    input_tensor = torch.randn(2, 3, d_model)
    
    # Track operation order
    operation_order = []
    
    def norm_hook(module, input_tensor, output_tensor):
        operation_order.append('norm')
        return output_tensor
    
    def ffn_hook(module, input_tensor, output_tensor):
        operation_order.append('ffn')
        return output_tensor
    
    # Register hooks
    model.norm.register_forward_hook(norm_hook)
    model.ffn.register_forward_hook(ffn_hook)
    
    # Forward pass
    output = model(input_tensor)
    
    # Verify operation order
    assert operation_order == ['norm', 'ffn'], "Operations not executed in correct order"
    
    # Verify residual connection
    assert output.shape == input_tensor.shape, "Output shape should match input shape"
    assert not torch.equal(output, input_tensor), "Output should not be identical to input"
    
    print("Test Passed: Forward pass operations are in correct order")


def main():
    from hw4lib.model import FeedForwardLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'FeedForwardLayer': [
                {
                    'func': lambda: test_sublayer_feedforward(FeedForwardLayer),
                    'description': 'Test the feedforward sublayer'
                }
            ]
        }
    )   

    framework.run_tests()  
    framework.summarize_results()

if __name__ == '__main__':
    main()